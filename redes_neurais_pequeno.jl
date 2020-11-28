### A Pluto.jl notebook ###
# v0.12.14

using Markdown
using InteractiveUtils

# ╔═╡ 96eb9ae0-30bc-11eb-1e22-890708244624
begin
	using RDatasets
	using MLJ
	using Flux
	import Random.seed!; seed!(123)
	using Plots
end	

# ╔═╡ 0d3bf19e-3102-11eb-2661-73574bc6552b
md"### O problema

url: https://archive.ics.uci.edu/ml/datasets/Iris/

Data Set Information:

This is perhaps the best known database to be found in the pattern recognition literature. Fisher's paper is a classic in the field and is referenced frequently to this day. (See Duda & Hart, for example.) The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other.

Predicted attribute: class of iris plant.

This is an exceedingly simple domain.

This data differs from the data presented in Fishers article (identified by Steve Chadwick, spchadwick '@' espeedaz.net ). The 35th sample should be: 4.9,3.1,1.5,0.2,Iris-setosa where the error is in the fourth feature. The 38th sample: 4.9,3.6,1.4,0.1,Iris-setosa where the errors are in the second and third features.


Attribute Information:

1. sepal length in cm
2. sepal width in cm
3. petal length in cm
4. petal width in cm
5. class:
-- Iris Setosa
-- Iris Versicolour
-- Iris Virginica"

# ╔═╡ e72f92ae-30c4-11eb-1940-6d50060c106d
iris = RDatasets.dataset("datasets", "iris");

# ╔═╡ 3dadcf40-3114-11eb-39db-7108c541c711
head(iris)

# ╔═╡ 8a2e0720-3102-11eb-3f44-7d2bc065e1ba
describe(iris,:mean,:std,:min,:max,:eltype)

# ╔═╡ f0538220-30c4-11eb-37ad-a125378947fe
y, X = unpack(iris, ==(:Species), colname -> true, rng=123);

# ╔═╡ 53d0da90-3116-11eb-06b3-a3c6ebc1a281
md"É interessante verificarmos as dimensõs dos nossos dados e também o número de classes. MLJFlux já identifica para você e ajusta a entrada e saída da rede. Entretanto, a arquitetura é simplória. Veremos mais a frente como podemos fazer a nossa arquitetura."

# ╔═╡ b00ae7b0-30e9-11eb-3649-41d9c676166c
size(X),unique(y)

# ╔═╡ fbad2992-3101-11eb-01c3-5f4c0a471575
md"### Criando o classificador"

# ╔═╡ 87101960-30c2-11eb-367f-dfeab5e2cffb
begin
	@load NeuralNetworkClassifier
	clf = NeuralNetworkClassifier()
end	

# ╔═╡ a65f7890-30c4-11eb-2c1f-ed94a0ec1243
mach = machine(clf, X, y)

# ╔═╡ ea5995c2-3101-11eb-36a3-25db3da1790b
md"### Particionando os dados 

Realizamos aqui o que conhecemos como hold-out. Particionamos em treino e teste, que é a forma mais simples de se fazer uma validação. Sabemos que existem outras no curso. No caso, deixei 70% para treino e 30% para teste. Como podem notar, a função gera índices qu correspondem às linhas dos exemplos. Ele não tira cópia dos dados."

# ╔═╡ 6d1c16a0-30c5-11eb-1e52-6f98c2648e77
train, test = partition(eachindex(y), 0.9, shuffle=true, rng=1234)

# ╔═╡ d572a200-3101-11eb-2190-4ff72e2dc603
md"### Treinando a Rede

Um conceito interessante do MLJ é o machine. Esta função faz uma mapeamento do preditor ao dado. Quando aplicamos machine, estamos gerando um nó num grafo de computação e um estado associado. Existem algmas vantagens neste paradigma como: associar um mesmo preditor a vários dados distintos que podem ser processados em diferentes recursos; evitar re execução de um mesmo preditor e um mesmo dado."

# ╔═╡ b827b20e-30c2-11eb-2229-f703a237dcc4
fit!(mach,rows=train)

# ╔═╡ bbef79c0-3115-11eb-3948-f3c4bf0be576
md" Aqui conseguimos ver a arquitetura da rede padrão do MLJFlux. Podemos alterar se quisermos."

# ╔═╡ 365eb290-30e2-11eb-3665-a3d1951a9fcc
fitted_params(mach).chain

# ╔═╡ a7500660-3101-11eb-1430-a5021429fdec
md"### Avaliando a qualidade no conjunto de teste

Aqui coloco algumas métricas já conhecidas de classificação para avaliarmos no conjunto de teste."

# ╔═╡ 87dd9c60-30df-11eb-0dd7-1976c80dd295
ŷ = predict_mode(mach, rows=test)

# ╔═╡ f92ad200-30e1-11eb-0168-35b7013a8ba5
cm = confusion_matrix(ŷ,y[test])

# ╔═╡ 05c4ae4e-30e2-11eb-0018-37d1d82e0cc4
accuracy(cm)

# ╔═╡ 0735f190-30e2-11eb-1b10-974050eb025d
matthews_correlation(cm)

# ╔═╡ 24a34c80-30f3-11eb-3224-9b3dbd304190
md"## Analisando a qualidade da arquitetura

Uma coisa comum na avaliação de redes é a análise da qualidade por época. Os resultados aqui são da predição a cada época. "

# ╔═╡ 384339e0-31ba-11eb-0d12-fbf707d43b44
md"#### Analisando a sensibilidade com relação as épocas

É importante descobrir o número ideal de épocas. Para isso, devemos registrar o erro por época."

# ╔═╡ 60da1820-30e2-11eb-3992-73afe463dc94
begin
	re = range(clf, :epochs, lower=1, upper=550, scale=:log10)
	curve = learning_curve(clf,X,y,
						   range=re,					                             resampling=Holdout(fraction_train=0.9,
			         shuffle=true,rng=123),#CV(nfolds=10,  shuffle=true, rng=123))
		             measure = cross_entropy)

end	


# ╔═╡ 4271ca40-31b4-11eb-2d56-2b521a23f1bd
plot(curve.parameter_values,
		   curve.measurements,
		   xlab=curve.parameter_name,
		   xscale=curve.parameter_scale,
		   ylab = "Cross Entropy",
		   label = "Validation Error")

# ╔═╡ 484c9a70-31ba-11eb-0b28-7bb96e703c3a
md"#### Analisando a sensibilidade com relação à taxa de aprendizado

Sabemos que a solução do método de aprendizado depende da taxa de aprendizado.
"

# ╔═╡ 04b46700-31b7-11eb-1702-a7af6b66681c
begin
	rl = range(clf, :lambda, lower=0.001, upper=1)
	curvel = learning_curve(clf,X,y,
						   range=rl,					                             resampling=Holdout(fraction_train=0.9,
			         shuffle=true,rng=123),#CV(nfolds=10,  shuffle=true, rng=123))
		             measure = cross_entropy)

end	


# ╔═╡ 1780b820-31b7-11eb-255d-b99cfb673820
plot(curvel.parameter_values,
		   curvel.measurements,
		   xlab=curvel.parameter_name,
		   xscale=curvel.parameter_scale,
		   ylab = "Cross Entropy",
		   label = "Validation Error")

# ╔═╡ 30647c00-31b7-11eb-3eed-3999ee53a4a9
clf.lambda

# ╔═╡ 65213270-31b8-11eb-3682-e51a60edd85d
clf.epochs

# ╔═╡ 92346140-31b5-11eb-0317-8f9a5589b0df
curve

# ╔═╡ c5d00cc0-31b5-11eb-0abf-a5eed9d49f11
clf.epochs=300

# ╔═╡ 5afe85c0-31ba-11eb-309d-1f526cf67d66
md"### Realizando uma validação cruzada 

O objetivo aqui é estimar com menor viés possível a qualidade da rede para a tarefa"

# ╔═╡ 8d6ff0d0-31b4-11eb-1860-ada5b3b6b84c
history= evaluate!(
    mach;
    resampling=CV(nfolds=10,  shuffle=true, rng=123),#Holdout(fraction_train=0.7, shuffle=true, rng=123),
    operation=predict_mode,
    measure=[accuracy, misclassification_rate],
    verbosity = 3
)

# ╔═╡ 102dfbb0-31b6-11eb-2019-c520f466d6a6
history

# ╔═╡ 1c0a34c0-30ef-11eb-26b0-b38e8f85eca6
md"### Criando a sua própria arquitetura de rede"

# ╔═╡ 17e44710-3116-11eb-0f52-2f6657afcead
md" Um ponto interessant é podermos modificar nossa arquitetura da rede e verificar se o desempenho melhora ou piora. No MLJFlux é preciso criar uma estrutura que será passada por argumento e nós sobrescrevemos a função build."

# ╔═╡ 6c58bf70-30ed-11eb-2fe4-111b91c5fdd2
begin
	mutable struct MyNetwork <: MLJFlux.Builder
		n1 :: Int
		n2 :: Int
	end

	function MLJFlux.build(nn::MyNetwork, n_in, n_out)
		return Chain(Dense(n_in, nn.n1,relu), 
			   Dense(nn.n1, nn.n2,relu), 
			   Dense(nn.n2, n_out))
	end
end	

# ╔═╡ 3503f1a0-30ef-11eb-0dc0-1bbcdf6478b7
myclf = NeuralNetworkClassifier(builder=MyNetwork(10,10))

# ╔═╡ 71a84080-31ac-11eb-034e-69fde3328410
mymach = machine(myclf, X, y)

# ╔═╡ 06495780-31ba-11eb-3e53-9732577482ef
md"### Analisando a sensibilidade com relação as épocas"

# ╔═╡ 813e9d90-30f4-11eb-361a-c1a55c36f13c
begin
	re1 = range(myclf, :epochs, lower=1, upper=250, scale=:log10)
	curvee1 = learning_curve(myclf,X,y,
						   range=re1,					                             resampling=Holdout(fraction_train=0.9,
			         shuffle=true,rng=123),#CV(nfolds=10,  shuffle=true, rng=123))
		             measure = cross_entropy)
end	


# ╔═╡ 27f2a5e0-31b4-11eb-1076-e5851c3482e1
plot(curvee1.parameter_values,
		   curvee1.measurements,
		   xlab=curvee1.parameter_name,
		   xscale=curvee1.parameter_scale,
		   ylab = "Cross Entropy",
	       label = "Error")

# ╔═╡ 15ef1440-31ba-11eb-0a5d-8d9578bf921f
md"### Realizando uma validação cruzada

O objetivo aqui é estimar com menor viés possível a qualidade da rede para a tarefa
"

# ╔═╡ ad02ba40-31b9-11eb-033e-2b06818b1dea
history2 = evaluate!(
    mymach;
    resampling=CV(nfolds=10,  shuffle=true, rng=123),#Holdout(fraction_train=0.7, shuffle=true, rng=123),
    operation=predict_mode,
    measure=[accuracy, misclassification_rate],
    verbosity = 3
)

# ╔═╡ ccafdbc0-31b9-11eb-2f03-07aaf5a96058
history2

# ╔═╡ f5af28a0-31b9-11eb-14e7-5794590639c4
md"### Um simples holdout"

# ╔═╡ 2e6caf2e-31ad-11eb-295a-29d5a6d93bc0
fit!(mymach,rows=train)

# ╔═╡ 9e552180-31ab-11eb-3713-275f29c2ce1d
ŷn = predict_mode(mymach, rows=test)

# ╔═╡ bc93e5ee-31ab-11eb-245e-ed426c077c4e
cmn = confusion_matrix(ŷn,y[test])

# ╔═╡ 4ebcee80-31ad-11eb-3d22-9b957c247948
accuracy(cmn)

# ╔═╡ Cell order:
# ╠═96eb9ae0-30bc-11eb-1e22-890708244624
# ╟─0d3bf19e-3102-11eb-2661-73574bc6552b
# ╠═e72f92ae-30c4-11eb-1940-6d50060c106d
# ╠═3dadcf40-3114-11eb-39db-7108c541c711
# ╠═8a2e0720-3102-11eb-3f44-7d2bc065e1ba
# ╠═f0538220-30c4-11eb-37ad-a125378947fe
# ╟─53d0da90-3116-11eb-06b3-a3c6ebc1a281
# ╠═b00ae7b0-30e9-11eb-3649-41d9c676166c
# ╟─fbad2992-3101-11eb-01c3-5f4c0a471575
# ╠═87101960-30c2-11eb-367f-dfeab5e2cffb
# ╠═a65f7890-30c4-11eb-2c1f-ed94a0ec1243
# ╟─ea5995c2-3101-11eb-36a3-25db3da1790b
# ╠═6d1c16a0-30c5-11eb-1e52-6f98c2648e77
# ╟─d572a200-3101-11eb-2190-4ff72e2dc603
# ╠═b827b20e-30c2-11eb-2229-f703a237dcc4
# ╟─bbef79c0-3115-11eb-3948-f3c4bf0be576
# ╠═365eb290-30e2-11eb-3665-a3d1951a9fcc
# ╟─a7500660-3101-11eb-1430-a5021429fdec
# ╠═87dd9c60-30df-11eb-0dd7-1976c80dd295
# ╠═f92ad200-30e1-11eb-0168-35b7013a8ba5
# ╠═05c4ae4e-30e2-11eb-0018-37d1d82e0cc4
# ╠═0735f190-30e2-11eb-1b10-974050eb025d
# ╠═24a34c80-30f3-11eb-3224-9b3dbd304190
# ╟─384339e0-31ba-11eb-0d12-fbf707d43b44
# ╠═60da1820-30e2-11eb-3992-73afe463dc94
# ╠═4271ca40-31b4-11eb-2d56-2b521a23f1bd
# ╟─484c9a70-31ba-11eb-0b28-7bb96e703c3a
# ╠═04b46700-31b7-11eb-1702-a7af6b66681c
# ╠═1780b820-31b7-11eb-255d-b99cfb673820
# ╠═30647c00-31b7-11eb-3eed-3999ee53a4a9
# ╠═65213270-31b8-11eb-3682-e51a60edd85d
# ╠═92346140-31b5-11eb-0317-8f9a5589b0df
# ╠═c5d00cc0-31b5-11eb-0abf-a5eed9d49f11
# ╟─5afe85c0-31ba-11eb-309d-1f526cf67d66
# ╠═8d6ff0d0-31b4-11eb-1860-ada5b3b6b84c
# ╠═102dfbb0-31b6-11eb-2019-c520f466d6a6
# ╟─1c0a34c0-30ef-11eb-26b0-b38e8f85eca6
# ╟─17e44710-3116-11eb-0f52-2f6657afcead
# ╠═6c58bf70-30ed-11eb-2fe4-111b91c5fdd2
# ╠═3503f1a0-30ef-11eb-0dc0-1bbcdf6478b7
# ╠═71a84080-31ac-11eb-034e-69fde3328410
# ╠═06495780-31ba-11eb-3e53-9732577482ef
# ╠═813e9d90-30f4-11eb-361a-c1a55c36f13c
# ╠═27f2a5e0-31b4-11eb-1076-e5851c3482e1
# ╟─15ef1440-31ba-11eb-0a5d-8d9578bf921f
# ╠═ad02ba40-31b9-11eb-033e-2b06818b1dea
# ╠═ccafdbc0-31b9-11eb-2f03-07aaf5a96058
# ╠═f5af28a0-31b9-11eb-14e7-5794590639c4
# ╠═2e6caf2e-31ad-11eb-295a-29d5a6d93bc0
# ╠═9e552180-31ab-11eb-3713-275f29c2ce1d
# ╠═bc93e5ee-31ab-11eb-245e-ed426c077c4e
# ╠═4ebcee80-31ad-11eb-3d22-9b957c247948
