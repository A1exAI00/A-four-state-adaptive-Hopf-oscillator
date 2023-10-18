#=
# TODO
=#

include("../src/oscillator_models.jl")
include("../src/time_series_tools.jl")
include("../src/misc_tools.jl")

using StaticArrays, CairoMakie

########################################################################

# Настройки генерируемого графика
PLOT_RES = (1000, 800)
PLOT_SAVING_DIR = "generated"; println(pwd())
PLOT_FILENAME = "02-four_state_cos"
PLOT_PX_PER_UNIT_PNG = 2

PRINT_PROGRESS = true

########################################################################

# Постоянные параметры системы
γ, μ, ε, η, N = 1.0, 1.0, 0.9, 0.9, 3 # 1.0, 1.0, 0.9, 0.9, 3
Ω_teach = [20.0, 50.0, 100.0]
A_teach = [1.0, 2.0, 5.0]
Φ_teach = [0π, 1π, 3π/2]
system_param = SA[γ, μ, ε, η, N, Ω_teach..., A_teach..., Φ_teach...]

# Начальные условия системы
x₀, y₀, α₀ = 1.0, 0.0, 0.0
ω₀_min, ω₀_max, N_ω₀ = 10, 100, 100
ω₀_range = range(ω₀_min, ω₀_max, N_ω₀) #U₀ = SA[x₀, y₀, ω₀]

# Время интегрирования
t₀, t₁ = 0.0, 2_000.0
t_SPAN = [t₀, t₁]

# Проверка параметров и начальных условий
println("system_param = $system_param")
#println("U₀ = $U₀")
@assert length(Ω_teach) == N "length of `ω_teach` is not equal `N`"
@assert length(A_teach) == N "length of `A_teach` is not equal `N`"
@assert length(Φ_teach) == N "length of `Φ_teach` is not equal `N`"

########################################################################

ω_final = zeros(length(ω₀_range))
α_final = zeros(length(ω₀_range))

# Поледовтельное интегрирование 
for (i, ω₀) in enumerate(ω₀_range)
    PRINT_PROGRESS && println("ω₀=$(round(ω₀, digits=5))")
    U₀ = SA[x₀, y₀, ω₀, α₀]
    solution = four_state_integrate(U₀, t_SPAN, system_param)
    ω_final[i] = mean_of_tail(solution[3,:], 0.9)
    α_final[i] = mean_of_tail(solution[4,:], 0.9)
end

########################################################################

fig = Figure(resolution=PLOT_RES)
ax_α_ω = Axis(fig[1,1], 
    title="АЧХ: α(ω)",
    xlabel="ω",
    ylabel="α")

# Axis
hlines!(ax_α_ω, 0.0, color=:black)
vlines!(ax_α_ω, 0, color=:black)

# solution 
hlines!.(ax_α_ω, A_teach, color=:black, linestyle=:dot)
vlines!.(ax_α_ω, Ω_teach, color=:black, linestyle=:dot)

# model solution
scatter!(ax_α_ω, ω_final, α_final, markersize=20)

#axislegend(ax_α_ω)

savingpath = joinpath(PLOT_SAVING_DIR, PLOT_FILENAME*"$(time_ns()).png")
save(savingpath, fig, px_per_unit=PLOT_PX_PER_UNIT_PNG)