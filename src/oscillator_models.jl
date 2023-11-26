#=
Базовые функции, необходимые для интегрирования модели 
адаптивного осциллятора Хопфа (Стюарта-Ландау) и адаптивного 
Central Pattern Generator (CPG) из статьи
"A four-state adaptive Hopf oscillator"
=#

########################################################################

#using Statistics: mean
using OrdinaryDiffEq, StaticArrays, Interpolations
include("../src/misc_tools.jl")

########################################################################

const RELTOL::Float64, ABSTOL::Float64 = 1e-5, 1e-5 # 1e-5, 1e-5
const MAXITERS::Int64 = Int(1e8)

# Different alg: 
# Rodas5P - for stiff systems; faild on automatic differentiation
# Tsit5 - 
const ALG = Tsit5

########################################################################

function original_HAFO_model(u, p, t)
    x, y, ω = u
    γ, μ, ε, N = p[1:4]
    N = Int(N)
    Ω_teach = p[0N+5:1N+4]
    A_teach = p[1N+5:2N+4]
    Φ_teach = p[2N+5:3N+4]

    r² = x^2 + y^2; r = sqrt(r²)
    Fₜ = sum(A_teach.*cos.(Ω_teach .* t .+ Φ_teach))

    return SA[
        γ*(μ-r²)*x - ω*y + ε*Fₜ, 
        γ*(μ-r²)*y + ω*x,
        -ε*Fₜ*y/r]
end

function original_HAFO_itp_model(u, p, t)
    x, y, ω = u
    γ, μ, ε, T_itp, P_teach_itp = p

    r² = x^2 + y^2; r = sqrt(r²)
    Fₜ = P_teach_itp(mod(t, T_itp))

    return SA[
        γ*(μ-r²)*x - ω*y + ε*Fₜ, 
        γ*(μ-r²)*y + ω*x,
        -ε*Fₜ*y/r]
end

function simplified_HAFO_model(u, p, t)
    x, y, ω = u
    γ, μ, ε, N = p[1:4]
    N = Int(N)
    Ω_teach = p[0N+5:1N+4]
    A_teach = p[1N+5:2N+4]
    Φ_teach = p[2N+5:3N+4]

    r² = x^2 + y^2
    Fₜ = sum(A_teach.*cos.(Ω_teach .* t .+ Φ_teach))

    return SA[
        γ*(μ-r²)*x - ω*y + ε*Fₜ, 
        γ*(μ-r²)*y + ω*x,
        -ε*Fₜ*y]
end

function simplified_HAFO_itp_model(u, p, t)
    x, y, ω = u
    γ, μ, ε, T_itp, P_teach_itp = p

    r² = x^2 + y^2
    Fₜ = P_teach_itp(mod(t, T_itp))

    return SA[
        γ*(μ-r²)*x - ω*y + ε*Fₜ, 
        γ*(μ-r²)*y + ω*x,
        -ε*Fₜ*y]
end

function four_state_model(u, p, t)
    x, y, ω, α = u
    γ, μ, ε, η, N = p[1:5]
    N = Int(N)
    Ω_teach = p[0N+6:1N+5]
    A_teach = p[1N+6:2N+5]
    Φ_teach = p[2N+6:3N+5]

    r² = x^2 + y^2
    Fₜ = sum(A_teach.*cos.(Ω_teach .* t .+ Φ_teach)) - α*x

    return SA[
        γ*(μ-r²)*x - ω*y + ε*Fₜ, 
        γ*(μ-r²)*y + ω*x,
        -ε*Fₜ*y,
        η*Fₜ*x]
end

function four_state_itp_model(u, p, t)
    x, y, ω, α = u
    γ, μ, ε, η, T_itp, P_teach_itp = p

    r² = x^2 + y^2
    Fₜ = P_teach_itp(mod(t,T_itp)) - α*x

    return SA[
        γ*(μ-r²)*x - ω*y + ε*Fₜ, 
        γ*(μ-r²)*y + ω*x,
        -ε*Fₜ*y,
        η*Fₜ*x]
end

########################################################################

function original_HAFO_integrate(U₀, t_span, param; reltol=RELTOL, abstol=ABSTOL, maxiters=MAXITERS, check_success=false)
    prob = ODEProblem(original_HAFO_model, U₀, t_span, param)
    sol = solve(prob, ALG(), reltol=reltol, abstol=abstol, maxiters=maxiters)
    return (check_success && sol.retcode!=:Success) ? NaN : sol
end

function original_HAFO_itp_integrate(U₀, t_span, param; reltol=RELTOL, abstol=ABSTOL, maxiters=MAXITERS, check_success=false)
    prob = ODEProblem(original_HAFO_itp_model, U₀, t_span, param)
    sol = solve(prob, ALG(), reltol=reltol, abstol=abstol, maxiters=maxiters)
    return (check_success && sol.retcode!=:Success) ? NaN : sol
end

function simplified_HAFO_integrate(U₀, t_span, param; reltol=RELTOL, abstol=ABSTOL, maxiters=MAXITERS, check_success=false)
    prob = ODEProblem(simplified_HAFO_model, U₀, t_span, param)
    sol = solve(prob, ALG(), reltol=reltol, abstol=abstol, maxiters=maxiters)
    return (check_success && sol.retcode!=:Success) ? NaN : sol
end

function simplified_HAFO_itp_integrate(U₀, t_span, param; reltol=RELTOL, abstol=ABSTOL, maxiters=MAXITERS, check_success=false)
    prob = ODEProblem(simplified_HAFO_itp_model, U₀, t_span, param)
    sol = solve(prob, ALG(), reltol=reltol, abstol=abstol, maxiters=maxiters)
    return (check_success && sol.retcode!=:Success) ? NaN : sol
end

function four_state_integrate(U₀, t_span, param; reltol=RELTOL, abstol=ABSTOL, maxiters=MAXITERS, check_success=false)
    prob = ODEProblem(four_state_model, U₀, t_span, param)
    sol = solve(prob, ALG(), reltol=reltol, abstol=abstol, maxiters=maxiters)
    return (check_success && sol.retcode!=:Success) ? NaN : sol
end

function four_state_itp_integrate(U₀, t_span, param; reltol=RELTOL, abstol=ABSTOL, maxiters=MAXITERS, check_success=false)
    prob = ODEProblem(four_state_itp_model, U₀, t_span, param)
    sol = solve(prob, ALG(), reltol=reltol, abstol=abstol, maxiters=maxiters)
    return (check_success && sol.retcode!=:Success) ? NaN : sol
end