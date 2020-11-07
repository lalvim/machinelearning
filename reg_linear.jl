### A Pluto.jl notebook ###
# v0.12.6

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
md" ### Hipótese e modelo linear"

# ╔═╡ 78d2c0e0-1f98-11eb-06bc-b51c3b4b9f35
md"A seguir descrevemos a hipótese que consiste da aplicação de um modelo (conjunto de parâmetros) em um exemplo. Note que a hipótese sempre se refere à aplicação de um único exemplo e não ao conjunto de dados todo. "

# ╔═╡ 2e7709e0-1ded-11eb-2abd-d770bf1afe8a
md"$H_{\theta}(X_i) = \sum_{j}^{m}{x_{i}^{(j)} \theta^{(j)}} = X_i\theta^{T}$"

# ╔═╡ b05f9930-1f97-11eb-0083-07ebd21313e0
md"Exemplificando o produto de vetores, nossa hipótese faria o seguinte cálculo para um exemplo $X_i = \begin{bmatrix} 1 & 500 \end{bmatrix}$ e um vetor de parâmetros $\theta^T = \begin{bmatrix}
    0.5 \\
    1.0
\end{bmatrix}$:"

# ╔═╡ cbf998e0-1f96-11eb-0a8b-a3bf631bfe02
md"e.g. $H_{\theta}(X_i) = \begin{bmatrix}
    1 & 500 
  \end{bmatrix}\begin{bmatrix}
    0.5 \\
    1.0
\end{bmatrix} = 500.5$"

# ╔═╡ 4cc76c80-1f98-11eb-0c70-29aec0b92ac4
md"Note que o resultado é um escalar, pois estamos estimando um valor contínuo (problema de regressão)."

# ╔═╡ c0b094a0-1f98-11eb-2efb-07f680447b12
md"### Implementação a hipótese"

# ╔═╡ be0deb50-1ded-11eb-06bb-33e94217e815
H(χ,θ) = χ*Θ'

# ╔═╡ 2f5a6500-1ded-11eb-11dd-bd88bf102b98
md" ### Função de custo"

# ╔═╡ 30972ac0-1ded-11eb-1007-1d3833dfc264

md"$J(\theta)= \frac{1}{2n}\sum_{i=1}^{n}{(H_{\theta}(X_i) - y_i)²}$"


# ╔═╡ bba6d590-1e1d-11eb-2df6-67bd6c5d4777
md" A seguir estou calculando a função de custo sem loop, apenas por matrizes.Desta maneira, geramos em uma multiplicação de matrizes, um vetor de predições com a mesma dimensão da saída. Após isto, subtraímos elemento a elemento. Após isto, elevamos cada elemento ao quadrado e depois tiramos uma média. Explicado na aula anterior."

# ╔═╡ 9837a3d0-1ded-11eb-3c49-17a45387e46e
J(Θ,χ,γ) = .5μ((χ*Θ' .- γ).^2)

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
	
	e₂  = J(θ,χ,γ) 
	e₁  = e₂ + .1
	#Δ₁,Δ₂ = e₁ - e₂ + 2ϵ,e₁ - e₂
	💻 = [[e₂ θ]]
	i   = 1
	while e₂ < e₁ && e₂ > ϵ && i < τ    
	#while Δ₁ > Δ₂  && e₂ > ϵ && i < τ     
		
		∇     = μ((χ*θ' .- γ).* χ,dims=1)
		θ    -= α .* ∇
		
		e₁,e₂ = e₂,J(θ,χ,γ)
		#Δ₁,Δ₂ = Δ₂,e₁ - e₂
		verbose && if (i%10==0) append!(💻,[[e₂ θ]]) end
		
		i += 1
	end
	θ,vcat(💻...)
end	

# ╔═╡ 1f0e51f0-1df4-11eb-0feb-b53552fa63c9
md"### Testes de unitários"

# ╔═╡ 58dd7720-1e09-11eb-1904-af5f9991ece5
md" Sempre que for desenvolver suas funções, realize uma bateria de testes para cada uma antes mesmo de executar todo o seu programa. Isso vai lhe poupar muito tempo para uma futura depuração de código. Assim, para cada função, faça pelo menos um teste. Elabore casos bons de entrada para suas funções de forma que cubra bem o espaço de possibilidades. Seus casos devem ser bem simples para que num futuro, se der erro, você consiga rapidamente resolver."

# ╔═╡ 259ac040-1e39-11eb-3670-f5c5519a1978
md"### Padronizando os Dados "

# ╔═╡ 246cc792-1e39-11eb-3c20-f1623d8ea737
md"O objetivo de padronizar os dados é acelerar a convergência do método. Aqui utilizaremos a padronização zscore. Entretanto, para amostras pequenas, a padronização não funciona muito bem."

# ╔═╡ c467ca00-1e08-11eb-3d41-75a38532652b
md"### Verificando a curva de erro"

# ╔═╡ d1a88a00-1f8f-11eb-36b7-f39dabee0657
md"### Veificando o comportamento dos parâmetros"

# ╔═╡ b405694e-1f8f-11eb-0fad-5d0c3527d206
md"### Verificando o modelo"

# ╔═╡ d497051e-1f94-11eb-2ccb-a7d3e52570d1
md"Como podemos observar, o gradiente jogou o θ₁ para zero (viés, responsável pela translação) e ajustou o θ₂ (responsável pela rotação). Cabe ressaltar que estamos trabalhando com os dados padronizados. É possível, no entanto, despadronizar e obter predições no campo dos números originais."

# ╔═╡ 37c52e40-1e48-11eb-1e13-7ba1bebfabc4
md"### Depuração "

# ╔═╡ 44997b82-1e48-11eb-1821-f7f3aecd8093
md" Costumo colocar alguma variáveis soltas para que eu veja se está tudo correto. "

# ╔═╡ a419b220-1e4b-11eb-1299-295bf39df3f4
md"### Solução direta por álgebra linear"

# ╔═╡ a39cb9f0-1e4b-11eb-296e-bf19e5438416
train(X,Y) = inv(X'X)X'Y

# ╔═╡ 628624c0-1dfa-11eb-2afb-137bf77669e1
begin
	Θ1 = [0. 4.] 
	χ =  [1 50.;1 60.;1 100.; 1 200.]
	γ =  [200  ;   240  ;   400; 800]
	@assert J(Θ1,χ,γ) == 0.0
	@assert train(Θ1,χ,γ)[1] == [0. 4.] # iniciando com a solução, o θ deve continuar igual
end	

# ╔═╡ 4c632fee-1e39-11eb-00c6-c3f2a0774005
begin 
	χₚ = hcat(ones((4,1)), zscore(χ[:,2:end])) # padronizamos só o que não é viés
	γₚ = zscore(γ)
end	

# ╔═╡ d8ab5330-1e47-11eb-06b3-af3d2ee97c7b
χₚ

# ╔═╡ 86be2920-1e48-11eb-10f1-6740a2459951
γₚ

# ╔═╡ 74604cb2-1e05-11eb-19ef-09fd04dc52e3
(θᵏ,r) = train([4 5],χₚ,γₚ,verbose=true,α = 1e-3,ϵ=1e-5,τ=1e5);

# ╔═╡ b7913600-1e0c-11eb-362e-0b4b9ef9d1aa
plot(1:length(r[:,1]),r[:,1],xlabel="iterações",ylabel="J(θ)",title="Análise de erros",label="Erro")

# ╔═╡ fa0bd460-1e23-11eb-131f-f31bec373c07
begin
	p = plot(1:length(r[:,2]),r[:,2],xlabel="iterações",ylabel="θ₁,θ₂",title="Análise de parâmetros",label="θ₁")
	plot!(p,1:length(r[:,3]),r[:,3],xlabel="iterações",label="θ₂")
end	

# ╔═╡ cc73d620-1f8f-11eb-357d-b19b50c19280
begin
	f(x) = θᵏ[1] + θᵏ[2]*x # o mesmo que H(X) lá do início
	plot(χₚ[:,2],f,xlabel="m² (padronizado)",ylabel="Preço (padronizado)",title="Análise do da predição do modelo",label="θ₁ + θ₂x")
	scatter!(χₚ[:,2],γₚ,label="Dados Coletados")
end	

# ╔═╡ 41505b10-1e34-11eb-06db-750eb3ab11cf
r[end,:]

# ╔═╡ 29b04780-1e44-11eb-3bcf-8b5f121d157f
J(θᵏ,χₚ,γₚ)

# ╔═╡ d0de29e0-1e36-11eb-398c-5f0b55ea281e
θᵏ

# ╔═╡ 70775a70-1e4c-11eb-3cec-11fd54e9b637
train(χₚ,γₚ) 

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
# ╟─b05f9930-1f97-11eb-0083-07ebd21313e0
# ╟─cbf998e0-1f96-11eb-0a8b-a3bf631bfe02
# ╟─4cc76c80-1f98-11eb-0c70-29aec0b92ac4
# ╟─c0b094a0-1f98-11eb-2efb-07f680447b12
# ╠═be0deb50-1ded-11eb-06bb-33e94217e815
# ╟─2f5a6500-1ded-11eb-11dd-bd88bf102b98
# ╟─30972ac0-1ded-11eb-1007-1d3833dfc264
# ╟─bba6d590-1e1d-11eb-2df6-67bd6c5d4777
# ╠═9837a3d0-1ded-11eb-3c49-17a45387e46e
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
# ╟─c467ca00-1e08-11eb-3d41-75a38532652b
# ╠═74604cb2-1e05-11eb-19ef-09fd04dc52e3
# ╟─b7913600-1e0c-11eb-362e-0b4b9ef9d1aa
# ╟─d1a88a00-1f8f-11eb-36b7-f39dabee0657
# ╟─fa0bd460-1e23-11eb-131f-f31bec373c07
# ╟─b405694e-1f8f-11eb-0fad-5d0c3527d206
# ╟─d497051e-1f94-11eb-2ccb-a7d3e52570d1
# ╠═cc73d620-1f8f-11eb-357d-b19b50c19280
# ╟─37c52e40-1e48-11eb-1e13-7ba1bebfabc4
# ╟─44997b82-1e48-11eb-1821-f7f3aecd8093
# ╠═41505b10-1e34-11eb-06db-750eb3ab11cf
# ╠═29b04780-1e44-11eb-3bcf-8b5f121d157f
# ╠═d0de29e0-1e36-11eb-398c-5f0b55ea281e
# ╠═d8ab5330-1e47-11eb-06b3-af3d2ee97c7b
# ╠═86be2920-1e48-11eb-10f1-6740a2459951
# ╟─a419b220-1e4b-11eb-1299-295bf39df3f4
# ╠═a39cb9f0-1e4b-11eb-296e-bf19e5438416
# ╠═70775a70-1e4c-11eb-3cec-11fd54e9b637
