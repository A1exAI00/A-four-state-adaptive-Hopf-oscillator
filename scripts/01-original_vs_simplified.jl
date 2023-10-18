#=
Обучение упрощенного адаптивного осциллятора Хопфа (HAFO).
Модель обучается на частоту внешнего сигнала
Результат: время обучения оригинальной системы имеет выраженную зависимость от параметра μ.
Время обучения упрощенной системы более устойчиво к изменениям этого параметра.
=#

include("../src/oscillator_models.jl")
include("../src/time_series_tools.jl")
include("../src/misc_tools.jl")

using StaticArrays, CairoMakie

########################################################################

# Настройки генерируемого графика
PLOT_RES = (1000, 800)
PLOT_SAVING_DIR = "generated"; println(pwd())
PLOT_FILENAME = "01-original_vs_simplified"
PLOT_PX_PER_UNIT_PNG = 2

########################################################################

# Постоянные параметры системы
γ, μ, ε, N = 1.0, 1.0, 0.5, 2 # 1.0, 1.0, 0.9, 2
Ω_teach = [20.0, 160.0]
A_teach = [1.0, 2.0]
Φ_teach = [0.0, 0.0]
system_param = SA[γ, μ, ε, N, Ω_teach..., A_teach..., Φ_teach...]

# Начальные условия системы
x₀, y₀ = 1.0, 0.0
ω₀_1, ω₀_2 = 60.0, 85.0
U₀_1 = SA[x₀, y₀, ω₀_1]
U₀_2 = SA[x₀, y₀, ω₀_2]

# Время интегрирования
t₀, t₁ = 0.0, 1500.0
t_SPAN = [t₀, t₁]

# Проверка параметров и начальных условий
println("system_param = $system_param")
#println("U₀ = $U₀")
@assert length(Ω_teach) == N "length of `ω_teach` is not equal `N`"
@assert length(A_teach) == N "length of `A_teach` is not equal `N`"
@assert length(Φ_teach) == N "length of `Φ_teach` is not equal `N`"

########################################################################

# Интегрирование оригинальной системы 
@time solution_o_1 = original_HAFO_integrate(U₀_1, t_SPAN, system_param)
@time solution_o_2 = original_HAFO_integrate(U₀_2, t_SPAN, system_param)

# Интегрирование упрощенной системы
@time solution_s_1 = simplified_HAFO_integrate(U₀_1, t_SPAN, system_param)
@time solution_s_2 = simplified_HAFO_integrate(U₀_2, t_SPAN, system_param)

#t_sol_1 = solution.t

########################################################################

fig = Figure(resolution=PLOT_RES)
ax_ω = Axis(fig[1,1], 
    title="ω(t), μ=$(μ)",
    xlabel="t",
    ylabel="ω")

hlines!(ax_ω, 0.0, color=:black)
hlines!.(ax_ω, Ω_teach, color=:black, linestyle=:dot)

lines!(ax_ω, solution_o_1.t, solution_o_1[3,:], label="original1")
lines!(ax_ω, solution_o_2.t, solution_o_2[3,:], label="original2")
lines!(ax_ω, solution_s_1.t, solution_s_1[3,:], label="simplified1")
lines!(ax_ω, solution_s_2.t, solution_s_2[3,:], label="simplified2")

axislegend(ax_ω)


savingpath = joinpath(PLOT_SAVING_DIR, PLOT_FILENAME*"$(time_ns()).png")
save(savingpath, fig, px_per_unit=PLOT_PX_PER_UNIT_PNG)