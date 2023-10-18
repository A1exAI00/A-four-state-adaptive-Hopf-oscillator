#=
# TODO
=#

include("../src/oscillator_models.jl")
include("../src/time_series_tools.jl")
include("../src/misc_tools.jl")

using StaticArrays, CairoMakie, Base.Threads

########################################################################

# Настройки генерируемого графика
PLOT_RES = (1000, 800)
PLOT_SAVING_DIR = "generated"; println(pwd())
PLOT_FILENAME = "03-four_state_square_wave"
PLOT_PX_PER_UNIT_PNG = 2

PRINT_PROGRESS = true
PRINT_ELAPSED_TIME = true

#println("N_THREADS = $(nthreads())")

########################################################################

# Постоянные параметры системы
γ, μ, ε, η, N = 1.0, 1.0, 0.9, 0.9, 13 # 1.0, 1.0, 0.9, 0.9, 3

# Интерполяция квадратной волны
Ω_itp, duty_cycle, A_square_wave, Nₜ, noise_aplitude  = 30.0, 0.1, 1.0, 1_000, 0.25
T_itp = 2π/Ω_itp
itp_param = (0.0, T_itp, Nₜ, T_itp, duty_cycle, A_square_wave, noise_aplitude)
P_teach_itp = generate_square_itp(itp_param)

system_param = (γ, μ, ε, η, T_itp, P_teach_itp)

# Начальные условия системы
x₀, y₀, α₀ = 1.0, 0.0, 0.0
ω₀_min, ω₀_max, N_ω₀ = Ω_itp-20.0, N*Ω_itp+20.0, 100
ω₀_range = range(ω₀_min, ω₀_max, N_ω₀) #U₀ = SA[x₀, y₀, ω₀]

# Время интегрирования
t₀, t₁ = 0.0, 10_000.0
t_SPAN = [t₀, t₁]

########################################################################

ω_learned = zeros(length(ω₀_range))
α_learned = zeros(length(ω₀_range))

t_start = time_ns()

# Поледовтельное интегрирование 
#@threads for i in eachindex(ω₀_range)
for i in eachindex(ω₀_range)
    ω₀ = ω₀_range[i]
    PRINT_PROGRESS && println("ω₀=$(round(ω₀, digits=5))")
    U₀ = SA[x₀, y₀, ω₀, α₀]
    solution = four_state_itp_integrate(U₀, t_SPAN, system_param)
    ω_learned[i] = mean_of_tail(solution[3,:], 0.9)
    α_learned[i] = mean_of_tail(solution[4,:], 0.9)
end

elapsed_time = elapsed_time_string(time_ns()-t_start)
PRINT_ELAPSED_TIME && println(elapsed_time)

########################################################################

# Аналитический спектр
ω_plot_range = range(0, maximum(ω_learned), 100)
α_plot_func(ω) = 2*A_square_wave*(-1)/(ω*π)*(cos(ω*π)-1)
α_plot = α_plot_func.(ω_plot_range/Ω_itp)

########################################################################

fig = Figure(resolution=PLOT_RES)
ax_α_ω = Axis(fig[1,1], 
    title="АЧХ: α(ω); амплитуда шума = $(noise_aplitude)",
    xlabel="ω",
    ylabel="α")

# Axis
hlines!(ax_α_ω, 0.0, color=:black)
vlines!(ax_α_ω, 0, color=:black)

# Аналитическое решение 
lines!(ax_α_ω, ω_plot_range, α_plot, color=:blue)
for i in 1:N
    vlines!(ax_α_ω, i*Ω_itp, color=:black, linestyle=:dot)
end

# model solution
scatter!(ax_α_ω, ω_learned, α_learned, markersize=20)

#axislegend(ax_α_ω)

savingpath = joinpath(PLOT_SAVING_DIR, PLOT_FILENAME*"$(time_ns()).png")
save(savingpath, fig, px_per_unit=PLOT_PX_PER_UNIT_PNG)