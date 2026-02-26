using RxInfer
using Distributions
using Statistics

# One-step scalar Bayesian fusion model:
# latent x with Gaussian prior, single Gaussian observation y.
@model function _scalar_fusion_model(y, m_prior, v_prior, v_obs)
    x ~ Normal(mean = m_prior, variance = v_prior)
    y ~ Normal(mean = x, variance = v_obs)
end


function _closed_form_scalar_update(y_obs::Float64, m_prior::Float64, v_prior::Float64, v_obs::Float64)
    denom = max(v_prior + v_obs, 1e-12)
    k = v_prior / denom
    m_post = m_prior + k * (y_obs - m_prior)
    v_post = max((1.0 - k) * v_prior, 1e-12)
    return m_post, v_post
end


function _rxinfer_scalar_update(y_obs::Float64, m_prior::Float64, v_prior::Float64, v_obs::Float64)
    used_rxinfer = false
    m_post = m_prior
    v_post = v_prior

    try
        result = infer(
            model = _scalar_fusion_model(
                m_prior = m_prior,
                v_prior = v_prior,
                v_obs = v_obs,
            ),
            data = (y = y_obs,),
            returnvars = (x = KeepLast(),),
        )
        qx = result.posteriors[:x]
        m_post = Float64(mean(qx))
        v_post = Float64(var(qx))
        if isfinite(m_post) && isfinite(v_post)
            used_rxinfer = true
        else
            m_post, v_post = _closed_form_scalar_update(y_obs, m_prior, v_prior, v_obs)
            used_rxinfer = false
        end
    catch
        m_post, v_post = _closed_form_scalar_update(y_obs, m_prior, v_prior, v_obs)
        used_rxinfer = false
    end

    return m_post, max(v_post, 1e-12), used_rxinfer
end


function rxinfer_belief_step(
    prev_ee_mean::AbstractVector,
    prev_obj_mean::AbstractVector,
    prev_target_mean::AbstractVector,
    prev_ee_cov_diag::AbstractVector,
    prev_obj_cov_diag::AbstractVector,
    prev_target_cov_diag::AbstractVector,
    ee_obs::AbstractVector,
    obj_obs::AbstractVector,
    target_obs::AbstractVector,
    ee_vel::AbstractVector,
    obj_vel::AbstractVector,
    target_vel::AbstractVector,
    dt::Float64,
    process_noise_ee::Float64,
    process_noise_obj::Float64,
    process_noise_target::Float64,
    obs_noise_ee::Float64,
    obs_noise_obj::Float64,
    obs_noise_target::Float64,
    min_variance::Float64,
)
    dt_clamped = clamp(Float64(dt), 0.0, 0.10)
    minv = max(Float64(min_variance), 1e-12)

    prev_ee_mean = Float64.(collect(prev_ee_mean))
    prev_obj_mean = Float64.(collect(prev_obj_mean))
    prev_target_mean = Float64.(collect(prev_target_mean))
    prev_ee_cov_diag = Float64.(collect(prev_ee_cov_diag))
    prev_obj_cov_diag = Float64.(collect(prev_obj_cov_diag))
    prev_target_cov_diag = Float64.(collect(prev_target_cov_diag))
    ee_obs = Float64.(collect(ee_obs))
    obj_obs = Float64.(collect(obj_obs))
    target_obs = Float64.(collect(target_obs))
    ee_vel = Float64.(collect(ee_vel))
    obj_vel = Float64.(collect(obj_vel))
    target_vel = Float64.(collect(target_vel))

    ee_mean = zeros(Float64, 3)
    obj_mean = zeros(Float64, 3)
    target_mean = zeros(Float64, 3)
    ee_cov_diag = zeros(Float64, 3)
    obj_cov_diag = zeros(Float64, 3)
    target_cov_diag = zeros(Float64, 3)
    used_all_rxinfer = true

    for i in 1:3
        # EE
        ee_prior_mean = prev_ee_mean[i] + ee_vel[i] * dt_clamped
        ee_prior_var = max(prev_ee_cov_diag[i] + process_noise_ee, minv)
        ee_obs_var = max(obs_noise_ee, minv)
        ee_m, ee_v, ee_used = _rxinfer_scalar_update(
            ee_obs[i],
            ee_prior_mean,
            ee_prior_var,
            ee_obs_var,
        )
        ee_mean[i] = ee_m
        ee_cov_diag[i] = ee_v
        used_all_rxinfer &= ee_used

        # Object-relative
        obj_prior_mean = prev_obj_mean[i] + obj_vel[i] * dt_clamped
        obj_prior_var = max(prev_obj_cov_diag[i] + process_noise_obj, minv)
        obj_obs_var = max(obs_noise_obj, minv)
        obj_m, obj_v, obj_used = _rxinfer_scalar_update(
            obj_obs[i],
            obj_prior_mean,
            obj_prior_var,
            obj_obs_var,
        )
        obj_mean[i] = obj_m
        obj_cov_diag[i] = obj_v
        used_all_rxinfer &= obj_used

        # Target-relative
        target_prior_mean = prev_target_mean[i] + target_vel[i] * dt_clamped
        target_prior_var = max(prev_target_cov_diag[i] + process_noise_target, minv)
        target_obs_var = max(obs_noise_target, minv)
        target_m, target_v, target_used = _rxinfer_scalar_update(
            target_obs[i],
            target_prior_mean,
            target_prior_var,
            target_obs_var,
        )
        target_mean[i] = target_m
        target_cov_diag[i] = target_v
        used_all_rxinfer &= target_used
    end

    backend_name = used_all_rxinfer ? "rxinfer" : "rxinfer_fallback"
    return (
        ee_mean = ee_mean,
        obj_mean = obj_mean,
        target_mean = target_mean,
        ee_cov_diag = ee_cov_diag,
        obj_cov_diag = obj_cov_diag,
        target_cov_diag = target_cov_diag,
        backend_name = backend_name,
    )
end
