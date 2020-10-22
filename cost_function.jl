### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 10c7a648-1488-11eb-0f9c-c3bfe711a449
md" ### Nomenclatura"

# ╔═╡ 10170496-1488-11eb-0882-df72e258d63c
md"Seja $X$ uma matriz com dimensões $n x m$ com $n$ linhas e $m$ colunas que contém nossos exemplos. $X_i^{(j)}$ corresponde ao elemento de $X$ em que $i$ é uma linha da matriz (exemplo) e $j$ uma coluna da matriz (atributo), cujo acesso na linguagem representamos por x[i,j]. Para pegarmos um vetor linha desta matriz (exemplo) representamos formalmente por $X_i$ e na linguagem por x[i,:]."

# ╔═╡ 3996c40e-1489-11eb-266a-95abbb9cd199
md"Seja $\theta$ um vetor linha com $m$ elementos tal que $m$ corresponde ao número de atributos mais um. $\theta = [\theta^{(0)},\theta^{(1)}, ...,\theta^{(m)}]$ em que cada elemento corresponde ao j-ésimo parâmetro associado ao j-ésimo atributo. Na nossa lingugaem acessamos um elemento por $\theta[i]$ e o vetor por $\theta$"

# ╔═╡ 391d5e2a-1489-11eb-15d4-77bf4828cfcd
md"Seja $y$ nosso vetor alvo em que representamos o i-ésimo elemento por $y_i$. Na nossa linguagem representamos por $y[i]$."

# ╔═╡ 794e1306-1487-11eb-203c-37766708b1a2
md" ### Modelo linear"

# ╔═╡ 7882a5fe-1487-11eb-3f45-fd54863cdfbd
md"$H_{\theta}(X_i) = \sum_{j}^{m}{X_{i}^{(j)} \theta^{(j)}} = X_i\theta$"

# ╔═╡ f6213c24-1486-11eb-0981-ad6addc2a8ef
md" ### Função de custo"

# ╔═╡ d8c3e0d8-1485-11eb-3eab-a11e3d31269f

md"$J(\theta)= \frac{1}{2m}\sum_{i=1}^{n}{(H_{\theta}(X_i) - y_i)²} = \frac{1}{2m}\sum_{i=1}^{n}{(X_i\theta - y_i)²} = J_2$"


# ╔═╡ be81b7b2-148b-11eb-222c-135499b083cf
md"que pode ser expandido ainda mais na forma: "

# ╔═╡ d16ba126-148b-11eb-388b-ed88f5dc2099
md"$= \frac{1}{2m}\sum_{i=1}^{n}{(\sum_{j=1}^{m}X_{i}^{(j)}\theta_{j} - y_i)²} = J_1$"

# ╔═╡ 1980e48e-148d-11eb-282a-155aa6620659
md"Podemos ainda calcular toda a função de custo apenas por operações de matrizes e vetores sem um laço de repetição explícito da seguinte forma:"

# ╔═╡ b3ed93fc-1406-11eb-34d4-31acb0c7d641
begin 
   import Statistics: mean
   import BenchmarkTools: @benchmark, @btime
   import StaticArrays: @SMatrix	
   μ = mean	
end

# ╔═╡ 35a275a0-1402-11eb-3ce4-fba2f21b24d6
function J₁(Θ,χ,γ)
     n,m = size(χ)
	 ∑ = 0.0
	 for i=1:n
		h = 0.0
		for j=1:m 
		   h += Θ[j] * χ[i,j]
		end
		∑ += (h - γ[i])^2
	 end
	 .5∑/n
end 

# ╔═╡ 1c4d435e-140d-11eb-1585-476688f1ec0c
function J₂(Θ,χ,γ)
     n,m = size(χ)
	 ∑ = 0.0
	 for (x,y) in zip(eachrow(χ),γ)
		∑ += (Θ*x .- y).^2 ...
	 end
	 .5∑/n
end 

# ╔═╡ 329221e4-1498-11eb-0569-d9bde9e62e1a
md" É possível ainda vetorizar todo o cálculo sem precisar fazer um loop explícito."

# ╔═╡ 19392bd6-1405-11eb-3022-bbaecf23c544
J₃(Θ,χ,γ) = .5μ((χ*Θ' .- γ).^2) 

# ╔═╡ 7b5e9f88-1498-11eb-2acf-a139d4c8e8fb
md" Abaixo temos um exemplo de teste da função de custo."

# ╔═╡ 48f06c34-1405-11eb-067e-cbe31a8e3cf3
begin
	Θ = @SMatrix [0. 4.] 
	χ = @SMatrix [1 50.;1 60.;1 100.; 1 200.]
	γ = @SMatrix [200  ;   240  ;   400; 800]
	J₁(Θ,χ,γ),J₂(Θ,χ,γ),J₃(Θ,χ,γ)
end	

# ╔═╡ 1c69de24-1499-11eb-1151-bbc0bf07a89c
md"### Medindo o tempo computacional"

# ╔═╡ 93db1528-1498-11eb-035d-efef99b051ed
md"Podemos medir o tempo de cada função a partir de um benchmark. É importante ressaltar que diversas otimizações podem ser testadas para melhorar o tempo. Uma forma seria trabalhar com a matriz $X$ transposta e $\theta$ como vetor coluna. Isto porque Julia representa uma matriz em ordem de coluna (como Matlab, R, Fortran, ...) devido à compatibilidade com libs de fortran dierentemente de outras linguagens como C, C++, ... Há ainda otimizações com vetores estáticos, macros como @inbounds, ... tudo isto pode acelerar ainda mais seu código."

# ╔═╡ cdfe994c-1415-11eb-01c0-13fd220aaa92
@benchmark J₁(Θ,χ,γ) seconds=1 gctrial=true
#@benchmark J₂(Θ,χ,γ) seconds=1 gctrial=true
#@benchmark J₃(Θ,χ,γ) seconds=1 gctrial=true

# ╔═╡ 7ee07dd4-149d-11eb-1f6d-2d565ee61a12
md" ### E se fosse você escolhesse os valores de $\theta$ ?"

# ╔═╡ 7fff1162-149d-11eb-2ec6-c3ad197a6d4c


# ╔═╡ 6aa7e186-14a2-11eb-287d-35f60c88cc0b
@bind θ₀ html"<input type='range' step='0.1' min='-10' max='10'>"

# ╔═╡ d357e744-14a2-11eb-2652-c35400acb3cf
@bind θ₁ html"<input type='range' step='0.1' min='-10' max='10'>"

# ╔═╡ 150cec5a-14aa-11eb-2be7-390b12287032
begin
	using Plots; 
	x = range(1,stop=300,length=300)
	n = length(x)
	y = zeros(n)
	for i=1:n 
		y[i] = θ₀ + θ₁*x[i]
		#y[i] = θ₀ + θ₁^2*x[i]
		#y[i] = θ₀ + θ₁*x[i]^2
		
	end
	plot(x,y,xlims = (0,300),ylims = (100,900),xlabel="Metros",ylabel="Preço",label="estimado");
	scatter!(χ[:,2],γ[:],xlims = (0,300),ylims = (100,900),label="exemplo");
end

# ╔═╡ 8ef513d8-14a2-11eb-3f6e-5b2f0a09363e
θ = [θ₀ θ₁]

# ╔═╡ 7c888c54-14a5-11eb-315f-3d32578eaf32
md"##### Erro : $(J₁([θ₀ θ₁],χ,γ))"

# ╔═╡ bfd7693a-14a5-11eb-3479-057e03ef43f4
md" ### Gráfico do modelo linear"

# ╔═╡ f17bfc22-14b3-11eb-319a-47522cda243c
md"### Curva de erro"

# ╔═╡ 3ac52d04-14b4-11eb-17cf-1b2f8531335d
begin
	e = []
	for i=-1:1:10
	    for j=1:1:10
		    append!(e,J₁([i j],χ,γ))
		end
	end
	println("e = ",e)
	scatter(1:1:10,e,xlabel="θ₁",ylabel="J",label="erro") # apenas com relação à θ₁. para fazer com θ₀ teríamos outro eixo e teríamos uma superfície. 

end

# ╔═╡ 7fa33e30-14ba-11eb-2543-379982e50d7e


# ╔═╡ Cell order:
# ╟─10c7a648-1488-11eb-0f9c-c3bfe711a449
# ╟─10170496-1488-11eb-0882-df72e258d63c
# ╟─3996c40e-1489-11eb-266a-95abbb9cd199
# ╟─391d5e2a-1489-11eb-15d4-77bf4828cfcd
# ╟─794e1306-1487-11eb-203c-37766708b1a2
# ╟─7882a5fe-1487-11eb-3f45-fd54863cdfbd
# ╟─f6213c24-1486-11eb-0981-ad6addc2a8ef
# ╟─d8c3e0d8-1485-11eb-3eab-a11e3d31269f
# ╟─be81b7b2-148b-11eb-222c-135499b083cf
# ╟─d16ba126-148b-11eb-388b-ed88f5dc2099
# ╟─1980e48e-148d-11eb-282a-155aa6620659
# ╠═b3ed93fc-1406-11eb-34d4-31acb0c7d641
# ╠═35a275a0-1402-11eb-3ce4-fba2f21b24d6
# ╠═1c4d435e-140d-11eb-1585-476688f1ec0c
# ╟─329221e4-1498-11eb-0569-d9bde9e62e1a
# ╠═19392bd6-1405-11eb-3022-bbaecf23c544
# ╟─7b5e9f88-1498-11eb-2acf-a139d4c8e8fb
# ╠═48f06c34-1405-11eb-067e-cbe31a8e3cf3
# ╟─1c69de24-1499-11eb-1151-bbc0bf07a89c
# ╟─93db1528-1498-11eb-035d-efef99b051ed
# ╠═cdfe994c-1415-11eb-01c0-13fd220aaa92
# ╠═7ee07dd4-149d-11eb-1f6d-2d565ee61a12
# ╟─7fff1162-149d-11eb-2ec6-c3ad197a6d4c
# ╠═6aa7e186-14a2-11eb-287d-35f60c88cc0b
# ╠═d357e744-14a2-11eb-2652-c35400acb3cf
# ╠═8ef513d8-14a2-11eb-3f6e-5b2f0a09363e
# ╠═7c888c54-14a5-11eb-315f-3d32578eaf32
# ╠═bfd7693a-14a5-11eb-3479-057e03ef43f4
# ╠═150cec5a-14aa-11eb-2be7-390b12287032
# ╠═f17bfc22-14b3-11eb-319a-47522cda243c
# ╟─3ac52d04-14b4-11eb-17cf-1b2f8531335d
# ╠═7fa33e30-14ba-11eb-2543-379982e50d7e
