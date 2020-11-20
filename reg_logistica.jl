### A Pluto.jl notebook ###
# v0.12.11

using Markdown
using InteractiveUtils

# ╔═╡ 8dbcf370-1dec-11eb-0a2d-e967ec1f2fde
md" ### Nomenclatura"

# ╔═╡ c4a8db60-1dec-11eb-0c39-73a8aac79092
md"Seja $X$ uma matriz com dimensões $n x m$ com $n$ linhas e $m$ colunas que contém nossos exemplos. $x_i^{(j)}$ corresponde ao elemento de $X$ em que $i$ é uma linha da matriz (exemplo) e $j$ uma coluna da matriz (atributo), cujo acesso na linguagem representamos por x[i,j]. Para pegarmos um vetor linha desta matriz (exemplo) representamos formalmente por $X_i$ e na linguagem por x[i,:]."

# ╔═╡ cd532e00-1dec-11eb-06d9-9b7c76e9c1ba
md"Seja $\theta$ um vetor linha com $m$ elementos tal que $m$ corresponde ao número de atributos mais um. $\theta = [\theta^{(0)},\theta^{(1)}, ...,\theta^{(m)}]$ em que cada elemento corresponde ao j-ésimo parâmetro associado ao j-ésimo atributo. Na nossa linguagem acessamos um elemento por $\theta[i]$ e o vetor por $\theta$"

# ╔═╡ dc365e10-1dec-11eb-0e4f-95624ee468ea
md"Seja $y$ nosso vetor alvo em que representamos o i-ésimo elemento por $y_i$. Na nossa linguagem representamos por $y[i]$."

# ╔═╡ 571a5ea0-1dee-11eb-0eaf-bf746a6faa86
md"### Algumas libs importantes usadas neste experimento"

# ╔═╡ 317bbe60-1ded-11eb-2faa-c3e47c0df176
begin 
   import Statistics: mean,std
   import BenchmarkTools: @benchmark, @btime
   import StaticArrays: @SMatrix
   import Plots: plot,plot!,scatter,scatter!	
   import StatsBase: zscore
   μ = mean	
end

# ╔═╡ f9af3a70-1dec-11eb-34f0-91b7fc591b87
md" ### Hipótese e modelo de regressão logística"

# ╔═╡ 78d2c0e0-1f98-11eb-06bc-b51c3b4b9f35
md"A seguir descrevemos a hipótese que consiste da aplicação de um modelo (conjunto de parâmetros) em um exemplo. Note que a hipótese sempre se refere à aplicação de um único exemplo e não ao conjunto de dados todo. Neste caso, aplicamos um modelo linear $z$ como expoente da função logística, cujo objetivo é delimitar a saída num intervalo entre 0 e 1."

# ╔═╡ 2e7709e0-1ded-11eb-2abd-d770bf1afe8a
md"$H_{\theta}(X_i) = \frac{1}{1 + e^{-z}} = \frac{1}{1 + e^{-X_i \theta^T }} = P(Y_i|X_i;\theta)$"

# ╔═╡ 9da10890-2aac-11eb-0069-15538564fb84
begin
	x = -10:0.1:10
	plot(x,1 ./(1 .+ ℯ.^(-x)),label="logit")
end	

# ╔═╡ b05f9930-1f97-11eb-0083-07ebd21313e0
md"Exemplificando o produto de vetores, nossa hipótese faria o seguinte cálculo para um exemplo $X_i = \begin{bmatrix} 1 & 2 \end{bmatrix}$ e um vetor de parâmetros $\theta^T = \begin{bmatrix}
    0.5 \\
    1.0
\end{bmatrix}$:"

# ╔═╡ cbf998e0-1f96-11eb-0a8b-a3bf631bfe02
md"e.g. $z = \begin{bmatrix}
    1 & 2 
  \end{bmatrix}\begin{bmatrix}
    0.5 \\
    1.0
\end{bmatrix} = 2.5$ então teremos $\frac{1}{1 + e^{-z}}$ = $(1 ./(1 .+ ℯ.^(-2.5)))"

# ╔═╡ 4cc76c80-1f98-11eb-0c70-29aec0b92ac4
md"Note que o resultado de $z$ é maior que 1. Entretanto, a função logística coloca no intervalo entre 0 e 1."

# ╔═╡ c0b094a0-1f98-11eb-2efb-07f680447b12
md"### Implementação a hipótese"

# ╔═╡ be0deb50-1ded-11eb-06bb-33e94217e815
H(χ,θ) = vec(1.0 ./(1.0 .+ ℯ .^(-χ*θ'))); #vec: garantir que seja um vetor coluna

# ╔═╡ 2f5a6500-1ded-11eb-11dd-bd88bf102b98
md" ### Função de custo"

# ╔═╡ 30972ac0-1ded-11eb-1007-1d3833dfc264

md"$J(\theta)= -\frac{1}{n}[ \sum_{i=1}^{n}{y_i log(P(Y_i|X_i;\theta)) + (1 - y_i )log(1 - P(Y_i|X_i;\theta))]  }$"


# ╔═╡ bba6d590-1e1d-11eb-2df6-67bd6c5d4777
md" A seguir estou calculando a função de custo sem loop, apenas por matrizes.Desta maneira, geramos em uma multiplicação de matrizes, um vetor de predições com a mesma dimensão da saída. Após isto, subtraímos elemento a elemento. Após isto, elevamos cada elemento ao quadrado e depois tiramos uma média. Explicado na aula anterior."

# ╔═╡ 9837a3d0-1ded-11eb-3c49-17a45387e46e
function J(py::Union{Array{Float64,1},Float64},γ::Union{Array{Float64,1},Float64}) 
	μ(-γ .* log.(2,py .+ 1e-4) .- (1 .- γ) .* (log.(2,1e-4+1 .- py)))
end	

# ╔═╡ 86e80760-2ac1-11eb-0326-7f2448320bc2
begin
	py  = collect(0.01:0.01:1)
	y0  = zeros(length(py))
	y1  = ones(length(py))
    plot(py,J.(py,y0),label="y=0",ylabel="J",xlabel="P(yᵢ)")
	plot!(py,J.(py,y1),label="y=1")
end	

# ╔═╡ 2ebaad22-1dee-11eb-2fb1-f51c49ec13fb
md"### Gradiente descendente"

# ╔═╡ d1cf31b0-1e17-11eb-22d1-916c9a331700
md"A seguir, mostramos o cálculo do gradiente. Como podemos notar, o expoente 2 desce e é eliminado pela constante 2 do denominador. Adicionalemente, vemos que é uma subtração do gradiente, pois estamos minimizando a função de custo. Por isto, estamos caminhando no sentido contrário ao do gradiente. É importante também ressaltar que o viés já está embutido na matriz $X$ na forma de uma coluna com valores 1. Desta forma, o cálculo do gradiente é o mesmo para o parâmetro do viés ($θ^{(0)}$) e os demais." 

# ╔═╡ cbf654c0-1dfa-11eb-0994-85a67ded79b0
md"$θ^{(j)} = θ^{(j)} - α \frac{1}{n} \sum_{i=1}^{n}{(H_θ(X_i) - y_i)x_{i}^{(j)}}$"

# ╔═╡ 46484b80-1e1d-11eb-2bdc-e50b24f6b530
md"Entretanto, podemos condensar esta fórmula usando notação de vetores e matrizes. Lembrando que $\theta$ é um vetor agora e os cálculos são feitos simultaneamente para cada eleemnto deste. Podemos observar que o índice da coluna some, pois são processadas automaticamente. A seguir, veja a fórmula:"

# ╔═╡ f1089b10-1e1d-11eb-120e-cdbd993d72dc
md"$θ = θ - α \frac{1}{n} \sum_{i=1}^{n}{(H_θ(X_i) - y_i) \circ X_{i}}$"

# ╔═╡ 79bc38d0-1e1f-11eb-066c-2929340c479e
md"Cabe observar que $\circ$ é o produto de hadamard."

# ╔═╡ f5d33090-1e1f-11eb-3f25-d10f2616803d
md"### Algoritmo "

# ╔═╡ 7a22eab0-1def-11eb-2672-495fb9cf19f8
function train(θ,χ,γ;verbose=false,α = 1e-3,ϵ=1e-5,τ=1e3)
	e₂  = J(H(χ,θ),γ) 
	e₁  = e₂ + .1
	#Δ₁,Δ₂ = e₁ - e₂ + 2ϵ,e₁ - e₂
	💻 = [[e₂ θ]]
	i   = 1
	#while e₂ > ϵ && i < τ
	while e₁ > e₂ && e₂ > ϵ && i < τ    
    #while Δ₁ > Δ₂  && e₂ > ϵ && i < τ     
		∇     = μ((H(χ,θ) .- γ).* χ,dims=1)
		θ    -= α .* ∇
		e₁,e₂ = e₂,J(H(χ,θ),γ)
		#Δ₁,Δ₂ = Δ₂,e₁ - e₂
		verbose && if (i%10==0) append!(💻,[[e₂ θ]]) end
		
		i += 1
	end
	
	θ,vcat(💻...)
end	

# ╔═╡ 1f0e51f0-1df4-11eb-0feb-b53552fa63c9
md"### Testes de unitários (função OU)"

# ╔═╡ 58dd7720-1e09-11eb-1904-af5f9991ece5
md" Sempre que for desenvolver suas funções, realize uma bateria de testes para cada uma antes mesmo de executar todo o seu programa. Isso vai lhe poupar muito tempo para uma futura depuração de código. Assim, para cada função, faça pelo menos um teste. Elabore casos bons de entrada para suas funções de forma que cubra bem o espaço de possibilidades. Seus casos devem ser bem simples para que num futuro, se der erro, você consiga rapidamente resolver."

# ╔═╡ 628624c0-1dfa-11eb-2afb-137bf77669e1
begin
	χ = [1.0 1.0 1.0; 
		1 0 1; 
		1 1 0; 
		1 0 0;
		1 0.3 0.4;
		1 0.5 0.6;
		1 0.8 0.4;
		1 0.4 0.1
	]
	γ = [1. ;1.; 1.; 0.; 0 ; 1 ; 1; 0]
	Θ = [0.0 0 0]
	@assert J(H(χ,Θ),γ) == 0.9997114898418766
	@assert H(χ,Θ) == [0.5 ; 0.5 ; 0.5;  0.5 ; 0.5 ; 0.5;  0.5;  0.5]
	@assert (H(χ,Θ) .- γ) == [-0.5;  -0.5;  -0.5;  0.5;  0.5;  -0.5;  -0.5;  0.5]
	
end	

# ╔═╡ 259ac040-1e39-11eb-3670-f5c5519a1978
md"### Padronizando os Dados "

# ╔═╡ 246cc792-1e39-11eb-3c20-f1623d8ea737
md"O objetivo de padronizar os dados é acelerar a convergência do método. Aqui utilizaremos a padronização zscore. Entretanto, para amostras pequenas, a padronização não funciona muito bem."

# ╔═╡ 4c632fee-1e39-11eb-00c6-c3f2a0774005
begin 
	χₚ = hcat(ones((8,1)), zscore(χ[:,2:end])) # padronizamos só o que não é viés
end	

# ╔═╡ c467ca00-1e08-11eb-3d41-75a38532652b
md"### Verificando a curva de erro"

# ╔═╡ 74604cb2-1e05-11eb-19ef-09fd04dc52e3
(θᵏ,r) = train([.1 .0 0],χ,γ,verbose=true;α = 0.5,ϵ=1e-2,τ=1e4);

# ╔═╡ b7913600-1e0c-11eb-362e-0b4b9ef9d1aa
plot(1:length(r[:,1]),r[:,1],xlabel="iterações",ylabel="J(θ)",title="Análise de erros",label="Erro")

# ╔═╡ d1a88a00-1f8f-11eb-36b7-f39dabee0657
md"### Veificando o comportamento dos parâmetros"

# ╔═╡ fa0bd460-1e23-11eb-131f-f31bec373c07
begin
	plot(1:length(r[:,2]),r[:,2],xlabel="iterações",ylabel="θ₁,θ₂,θ₃",title="Análise de parâmetros",label="θ₁")
	plot!(1:length(r[:,3]),r[:,3],xlabel="iterações",label="θ₂")
	plot!(1:length(r[:,4]),r[:,4],xlabel="iterações",label="θ₃")
end	

# ╔═╡ b405694e-1f8f-11eb-0fad-5d0c3527d206
md"### Testando as Classificações"

# ╔═╡ d497051e-1f94-11eb-2ccb-a7d3e52570d1
md"Aqui, definiremos uma regra para classificação."

# ╔═╡ cc73d620-1f8f-11eb-357d-b19b50c19280
function classify(X::Array{Float64,2},θ::Array{Float64,2}) 
	H(X,θ)... > .5 ? 1 : 0
end	

# ╔═╡ 803ccbc0-2b53-11eb-1399-cf96cbb67727
classify([1 0.5 0.4],θᵏ)

# ╔═╡ 37c52e40-1e48-11eb-1e13-7ba1bebfabc4
md"### Depuração "

# ╔═╡ 44997b82-1e48-11eb-1821-f7f3aecd8093
md" Costumo colocar alguma variáveis soltas para que eu veja se está tudo correto. "

# ╔═╡ 41505b10-1e34-11eb-06db-750eb3ab11cf
r[end,:]

# ╔═╡ 29b04780-1e44-11eb-3bcf-8b5f121d157f
J(H(θᵏ,χ),	γ) 

# ╔═╡ f81b6012-2b4a-11eb-173a-1bf763fbd71d
H(θᵏ,χ)

# ╔═╡ d0de29e0-1e36-11eb-398c-5f0b55ea281e
θᵏ

# ╔═╡ d8ab5330-1e47-11eb-06b3-af3d2ee97c7b
χₚ

# ╔═╡ Cell order:
# ╟─8dbcf370-1dec-11eb-0a2d-e967ec1f2fde
# ╟─c4a8db60-1dec-11eb-0c39-73a8aac79092
# ╟─cd532e00-1dec-11eb-06d9-9b7c76e9c1ba
# ╟─dc365e10-1dec-11eb-0e4f-95624ee468ea
# ╟─571a5ea0-1dee-11eb-0eaf-bf746a6faa86
# ╠═317bbe60-1ded-11eb-2faa-c3e47c0df176
# ╟─f9af3a70-1dec-11eb-34f0-91b7fc591b87
# ╟─78d2c0e0-1f98-11eb-06bc-b51c3b4b9f35
# ╟─2e7709e0-1ded-11eb-2abd-d770bf1afe8a
# ╠═9da10890-2aac-11eb-0069-15538564fb84
# ╟─b05f9930-1f97-11eb-0083-07ebd21313e0
# ╟─cbf998e0-1f96-11eb-0a8b-a3bf631bfe02
# ╟─4cc76c80-1f98-11eb-0c70-29aec0b92ac4
# ╟─c0b094a0-1f98-11eb-2efb-07f680447b12
# ╠═be0deb50-1ded-11eb-06bb-33e94217e815
# ╟─2f5a6500-1ded-11eb-11dd-bd88bf102b98
# ╟─30972ac0-1ded-11eb-1007-1d3833dfc264
# ╟─bba6d590-1e1d-11eb-2df6-67bd6c5d4777
# ╠═9837a3d0-1ded-11eb-3c49-17a45387e46e
# ╠═86e80760-2ac1-11eb-0326-7f2448320bc2
# ╟─2ebaad22-1dee-11eb-2fb1-f51c49ec13fb
# ╟─d1cf31b0-1e17-11eb-22d1-916c9a331700
# ╟─cbf654c0-1dfa-11eb-0994-85a67ded79b0
# ╟─46484b80-1e1d-11eb-2bdc-e50b24f6b530
# ╟─f1089b10-1e1d-11eb-120e-cdbd993d72dc
# ╟─79bc38d0-1e1f-11eb-066c-2929340c479e
# ╟─f5d33090-1e1f-11eb-3f25-d10f2616803d
# ╠═7a22eab0-1def-11eb-2672-495fb9cf19f8
# ╟─1f0e51f0-1df4-11eb-0feb-b53552fa63c9
# ╟─58dd7720-1e09-11eb-1904-af5f9991ece5
# ╠═628624c0-1dfa-11eb-2afb-137bf77669e1
# ╟─259ac040-1e39-11eb-3670-f5c5519a1978
# ╟─246cc792-1e39-11eb-3c20-f1623d8ea737
# ╠═4c632fee-1e39-11eb-00c6-c3f2a0774005
# ╠═c467ca00-1e08-11eb-3d41-75a38532652b
# ╠═74604cb2-1e05-11eb-19ef-09fd04dc52e3
# ╠═b7913600-1e0c-11eb-362e-0b4b9ef9d1aa
# ╟─d1a88a00-1f8f-11eb-36b7-f39dabee0657
# ╠═fa0bd460-1e23-11eb-131f-f31bec373c07
# ╟─b405694e-1f8f-11eb-0fad-5d0c3527d206
# ╠═d497051e-1f94-11eb-2ccb-a7d3e52570d1
# ╠═cc73d620-1f8f-11eb-357d-b19b50c19280
# ╠═803ccbc0-2b53-11eb-1399-cf96cbb67727
# ╟─37c52e40-1e48-11eb-1e13-7ba1bebfabc4
# ╟─44997b82-1e48-11eb-1821-f7f3aecd8093
# ╠═41505b10-1e34-11eb-06db-750eb3ab11cf
# ╠═29b04780-1e44-11eb-3bcf-8b5f121d157f
# ╠═f81b6012-2b4a-11eb-173a-1bf763fbd71d
# ╠═d0de29e0-1e36-11eb-398c-5f0b55ea281e
# ╠═d8ab5330-1e47-11eb-06b3-af3d2ee97c7b
