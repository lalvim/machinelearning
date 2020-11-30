### A Pluto.jl notebook ###
# v0.12.14

using Markdown
using InteractiveUtils

# ╔═╡ c9762a40-3119-11eb-021f-3526881bee53
begin
   using MLJ, MLJFlux, Flux
   import Random.seed!; seed!(123)
   using Plots	
end	

# ╔═╡ ffc72f20-3184-11eb-0f84-f57d92c16b65
md"### Aqui construímos nosso próprio modelo usando MLJ+Flux.

É necessário a criação de uma estrutura que herda de um builder. Adicionalmente, devemos sobrecarregar a função build, pois esta será chamada internamente."

# ╔═╡ 563a4730-311b-11eb-0b9f-45f44f995aa6
begin

	mutable struct MNISTBuilder <: MLJFlux.Builder
		n_hidden::Int
	end

	function MLJFlux.build(builder::MNISTBuilder, (n1, n2), m, n_channels)
		return Chain(
			flatten,
			Dense(n1 * n2, builder.n_hidden, relu),
			Dense(builder.n_hidden, m),
		)
	end	
	
end	

# ╔═╡ c7f7cf90-3185-11eb-0f94-f5e182753280
md"### Instanciando Classificador"

# ╔═╡ 5c511db0-311b-11eb-3eec-2fb8f22a5a53
begin
	@load ImageClassifier
	clf = ImageClassifier(;
		builder=MNISTBuilder(128),
		optimiser=ADAM(0.001),
		loss=Flux.crossentropy,
		epochs=6,
		batch_size=128,
	)
end	

# ╔═╡ aa9142ae-3185-11eb-22a3-f39504976138
md"### O problema

url: http://yann.lecun.com/exdb/mnist/

The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.
It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting."

# ╔═╡ 5b631430-311b-11eb-1388-6f434e9cc354
begin
	X = Flux.Data.MNIST.images()
	y = Flux.Data.MNIST.labels()
	y = coerce(y, Multiclass)
end	

# ╔═╡ 130f6240-311d-11eb-0efd-7b4822376ef8
X[100]

# ╔═╡ e40bc4c0-3185-11eb-0794-63633f2c5269
md"### Particionando os dados 

Realizamos aqui o que conhecemos como hold-out. Particionamos em treino e teste, que é a forma mais simples de se fazer uma validação. Sabemos que existem outras no curso. No caso, deixei 70% para treino e 30% para teste. Como podem notar, a função gera índices qu correspondem às linhas dos exemplos. Ele não tira cópia dos dados."

# ╔═╡ fa3a71d0-3124-11eb-3b83-014b67169868
train, test = partition(eachindex(y), 0.01, shuffle=true, rng=1234)

# ╔═╡ d6fd7bc0-3185-11eb-13c0-7381eb7a588a
md"### Treinando a Rede

Um conceito interessante do MLJ é o machine. Esta função faz uma mapeamento do preditor ao dado. Quando aplicamos machine, estamos gerando um nó num grafo de computação e um estado associado. Existem algmas vantagens neste paradigma como: associar um mesmo preditor a vários dados distintos que podem ser processados em diferentes recursos; evitar re execução de um mesmo preditor e um mesmo dado."

# ╔═╡ 5d2b5110-311b-11eb-3dc3-0b7fdfd060fd
mach = machine(clf, X, y)

# ╔═╡ 177cb6c2-3186-11eb-2d73-39f8a0ac039b
fit!(mach,rows=train)

# ╔═╡ 3d71d0e0-3186-11eb-3a9c-d17db573ba54
md"### Predição 

Aqui usamos o predict_mode para pegar as previsões discretas sem as probabilidades. às vezes é preciso usar apenas o predict para pegar ambas as classes e probabilidades, pois certas métricas necessitam"

# ╔═╡ c5fd5630-3124-11eb-2844-7fc187c8a99b
ŷ = predict_mode(mach, rows=test)

# ╔═╡ d0212970-3124-11eb-2041-91a20d4349a2
cm = confusion_matrix(ŷ,y[test])

# ╔═╡ d1e7b8a0-3124-11eb-254e-83e4e57c4982
accuracy(cm)

# ╔═╡ dfab9600-3124-11eb-2459-6f59178a5d84
balanced_accuracy(ŷ,y[test])

# ╔═╡ 9d2bb6ae-31ed-11eb-2878-074b2adae35e
md"### Encontrando o melhor modelo"

# ╔═╡ 2e03a3a2-31ee-11eb-03eb-35d112cba7a6
clf.builder.n_hidden

# ╔═╡ a894b420-31ed-11eb-2de1-6b5a80d19ea9
begin

	# defining hyperparams for tunning
	r1 = range(clf, :lambda, lower=0.001, upper=1.0)
	r2 = range(clf, :epochs, lower=10, upper=300)
	r3 = range(clf, :(builder.n_hidden), lower=64, upper=256)

	# attaching tune
	self_tuning_ann_model = TunedModel(model        =  clf,
									   resampling   = CV(nfolds = 3),
									   tuning       = RandomSearch(),
									   range        = [r1,r2,r3],
									   measure      = accuracy,
									   operation    = predict_mode,
									   n=8)

	
end	

# ╔═╡ bd1883d0-31ee-11eb-0929-9f9e106252e9
begin
    # putting into the machine
	self_tuning_ann = machine(self_tuning_ann_model, X, y)

	# fitting with tunning
	fit!(self_tuning_ann,rows=train)
end

# ╔═╡ 2629c250-329c-11eb-332d-7debeb62dabe
fitted_params(self_tuning_ann)

# ╔═╡ 7f796040-329c-11eb-0cba-2f3c6648a92e
report(self_tuning_ann)

# ╔═╡ 7b4d6320-3186-11eb-3228-616d0e73b93c
md"### Aplicando o melhor modelo

Depois que testamos combinações com variações de parâmetros, vamos aplicar o modelo aprendido no conjunto de teste."

# ╔═╡ 5e5eef12-311b-11eb-23a9-6b715d078655
ŷb = predict_mode(self_tuning_ann,rows=test)

# ╔═╡ 8238a550-32b2-11eb-1d18-ed7f4b004b27
cmb = confusion_matrix(ŷb,y[test])

# ╔═╡ b35e47c0-32b2-11eb-1fce-cf420134aed0
accuracy(cmb)

# ╔═╡ b9377900-32b2-11eb-3478-e11f07450ec5
balanced_accuracy(ŷb,y[test])

# ╔═╡ Cell order:
# ╠═c9762a40-3119-11eb-021f-3526881bee53
# ╟─ffc72f20-3184-11eb-0f84-f57d92c16b65
# ╠═563a4730-311b-11eb-0b9f-45f44f995aa6
# ╟─c7f7cf90-3185-11eb-0f94-f5e182753280
# ╠═5c511db0-311b-11eb-3eec-2fb8f22a5a53
# ╟─aa9142ae-3185-11eb-22a3-f39504976138
# ╠═5b631430-311b-11eb-1388-6f434e9cc354
# ╠═130f6240-311d-11eb-0efd-7b4822376ef8
# ╟─e40bc4c0-3185-11eb-0794-63633f2c5269
# ╠═fa3a71d0-3124-11eb-3b83-014b67169868
# ╟─d6fd7bc0-3185-11eb-13c0-7381eb7a588a
# ╠═5d2b5110-311b-11eb-3dc3-0b7fdfd060fd
# ╠═177cb6c2-3186-11eb-2d73-39f8a0ac039b
# ╟─3d71d0e0-3186-11eb-3a9c-d17db573ba54
# ╠═c5fd5630-3124-11eb-2844-7fc187c8a99b
# ╠═d0212970-3124-11eb-2041-91a20d4349a2
# ╠═d1e7b8a0-3124-11eb-254e-83e4e57c4982
# ╠═dfab9600-3124-11eb-2459-6f59178a5d84
# ╟─9d2bb6ae-31ed-11eb-2878-074b2adae35e
# ╠═2e03a3a2-31ee-11eb-03eb-35d112cba7a6
# ╠═a894b420-31ed-11eb-2de1-6b5a80d19ea9
# ╠═bd1883d0-31ee-11eb-0929-9f9e106252e9
# ╠═2629c250-329c-11eb-332d-7debeb62dabe
# ╠═7f796040-329c-11eb-0cba-2f3c6648a92e
# ╟─7b4d6320-3186-11eb-3228-616d0e73b93c
# ╠═5e5eef12-311b-11eb-23a9-6b715d078655
# ╠═8238a550-32b2-11eb-1d18-ed7f4b004b27
# ╠═b35e47c0-32b2-11eb-1fce-cf420134aed0
# ╠═b9377900-32b2-11eb-3478-e11f07450ec5
