### A Pluto.jl notebook ###
# v0.12.10

using Markdown
using InteractiveUtils

# ╔═╡ 61dc86f0-238f-11eb-054f-41a45b78e94d
begin
	using  MLJ
	import RDatasets: dataset
    import DataFrames: describe, select, Not, rename!,head
	import Plots: plot,plot!,scatter,scatter!,histogram,histogram!	
end	

# ╔═╡ ff5bd480-29c0-11eb-3112-cb35c17acc60
md" ### Libs"

# ╔═╡ 8e226310-29c0-11eb-1093-f95e53c029b6
md"### Carga de Dados"

# ╔═╡ 114c91a0-2a90-11eb-365b-4d360bb51ebe
md" É interessante sempre mostrarmos os primeiros exemplos dos dados para ter uma ideia dos valores. Entretanto, como visto no curso, há várias análises possíveis para se fazer como identificar nulos, outliers, ... Tudo isto, é bom ser tratado."

# ╔═╡ b6faf1b0-294a-11eb-38ae-0924c4943d81
begin
	boston = dataset("MASS", "Boston")
	head(boston)
end	

# ╔═╡ a2efe1c0-29c8-11eb-287d-49b90e597876
md"### Sobre o Problema

url: https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html
"

# ╔═╡ a04c5f12-29d3-11eb-12a6-3f1b8acc88dc
md"#### Atributos"

# ╔═╡ a417e700-29c8-11eb-3ee5-6d2c302d95ac
md"There are 14 attributes in each case of the dataset. They are:

CRIM - per capita crime rate by town

ZN - proportion of residential land zoned for lots over 25,000 sq.ft.

INDUS - proportion of non-retail business acres per town.

CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)

NOX - nitric oxides concentration (parts per 10 million)

RM - average number of rooms per dwelling

AGE - proportion of owner-occupied units built prior to 1940

DIS - weighted distances to five Boston employment centres

RAD - index of accessibility to radial highways

TAX - full-value property-tax rate per $10,000

PTRATIO - pupil-teacher ratio by town

B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town

LSTAT - % lower status of the population

MEDV - Median value of owner-occupied homes in $1000's"

# ╔═╡ 9a810b20-29c0-11eb-1ca3-d155108c6e52
md"### Estatísticas dos Dados"

# ╔═╡ 91187080-2a8f-11eb-3001-1973d885e3f0
md"É sempre bom tiramos umas estatísticas dos dados para entender melhor o comportamento das variáveis do nosso problema. Qual a variância de cada variável? Elas estão na mesma ordem de grandeza? Máixmo e minímo ?"

# ╔═╡ 2033e842-294f-11eb-2b38-1969d7440b88
describe(boston, :mean, :std, :min, :max, :eltype)

# ╔═╡ 2f06c4a0-294f-11eb-2cb1-2da452f98b0f
data = coerce(boston, autotype(boston, :discrete_to_continuous));

# ╔═╡ d2a9dfe0-29c0-11eb-36a9-e7e3c1d17717
md"### Selecionando Dados"

# ╔═╡ 49b99b50-2a90-11eb-2b54-2704b914334d
md"É nesta fase que você vai escolher seu target e vai selecionar os atributos para trabalhar."

# ╔═╡ 47a80140-294f-11eb-2880-ef7cfb0a8997
begin
	y = data.MedV;
    X = select(data, Not(:MedV));
end	

# ╔═╡ e3d83ea0-29a3-11eb-10f5-b5f94102e1f1
md"### Verificando Algoritmos Disponíveis"

# ╔═╡ 60083560-2a90-11eb-373f-3b8a1ba538a0
md"Essa é uma característica bem legal do MLJ. Como nosso dados possuem atributos numéricos e o target numérico, ele me mostra todos os algoritmos que conseguem rodar para este tipo de dado."

# ╔═╡ 868b73b0-29a9-11eb-0ef8-971446a3c919
models(matching(X, y))

# ╔═╡ 87cedb30-2a90-11eb-26dd-6162d49edd88
md"Abaixo realizamos uma carga no nosso regressor linear."

# ╔═╡ 6b75f5a0-294f-11eb-0835-cfe4a412cffe
begin
	@load LinearRegressor pkg=MLJLinearModels	
	mdl = LinearRegressor();
end	

# ╔═╡ 9abff260-2a90-11eb-026f-59a85ec6b64c
md"O MLJ nos permite montar um pipeline. Pipeline é uma forma de montar uma sequência de operações poupando-lhe código. No nosso caso, queremos padronizar X e y antes de realizarmos o treino do regressor. Podemos fazer isso em uma única linha."

# ╔═╡ fcde2920-29a6-11eb-37f7-4befad3ee3d2
predictor = @pipeline Standardizer mdl target=Standardizer

# ╔═╡ 323d2620-29c0-11eb-258a-4703b4f5a0d2
md"### Partição dos Dados"

# ╔═╡ e57688a2-2a90-11eb-2272-3d9898bf4375
md"Realizamos aqui o que conhecemos como hold-out. Particionamos em treino e teste, que é a forma mais simples de se fazer uma validação. Sabemos que existem outras no curso. No caso, deixei 70% para treino e 30% para teste. Como podem notar, a função gera índices qu correspondem às linhas dos exemplos. Ele não tira cópia dos dados."

# ╔═╡ fca39122-29a6-11eb-3b25-c7f0df7fc68f
train, test  = partition(eachindex(y), 0.7, shuffle=true)

# ╔═╡ 287ae7e0-2a91-11eb-0d5b-b138443bdc29
md"Um conceito interessante do MLJ é o machine. Esta função faz uma mapeamento do preditor ao dado. Quando aplicamos machine, estamos gerando um nó num grafo de computação e um estado associado. Existem algmas vantagens neste paradigma como: associar um mesmo preditor a vários dados distintos que podem ser processados em diferentes recursos; evitar re execução de um mesmo preditor e um mesmo dado."

# ╔═╡ fc474050-29a6-11eb-1fbd-87a88d38525a
pred_machine = machine(predictor, X, y);

# ╔═╡ 4449f8d0-29b5-11eb-3f83-c352d034ecb2
md"### Treino da regressão linear "

# ╔═╡ 91918630-2a91-11eb-1113-43b18086f208
md" Aqui temos enfim, o treino do nosso regressor para o data X, apenas no conjunto de treino. Isto porque estamos passando os índices do exemplos."

# ╔═╡ 0de5e6e0-29a7-11eb-0558-2bb10a5fe1a5
fit!(pred_machine, rows=train);

# ╔═╡ 50f36580-29b5-11eb-06f2-cf4b9e142979
md"### Análise da Predição"

# ╔═╡ ef445730-2a91-11eb-1552-c3e7d6e5ff34
md"Aqui analisamos cada predição e cada target. Este tipo de análise é mais apropriada quando nosso conjunto de dados possui uma relação de ordem temporal entre os exemplos (que não é o caso). Entretanto, é uma análise boa a se fazer. Podemos notar que há um atendência de subestimar os valores mais altos do que valores mais baixos. Você pode analisar o que pode estar gerando dificldade nisto analisando alguns exemplos individualmente."

# ╔═╡ 152f7100-29a7-11eb-2300-735ca72b15c3
ŷ = predict(pred_machine, rows=test)

# ╔═╡ c29c1380-29b5-11eb-254c-2d026e15f903
begin
	plot(ŷ,label="Target")
	scatter!(y[test],label="Label")
	
end	

# ╔═╡ 89d06380-29bf-11eb-0d5b-4533c94745d5
md" Distribuição da Predição"

# ╔═╡ 8b3ac8f0-29bf-11eb-274e-f7b07eb35e7c
begin
   histogram(ŷ,label="Target")
end	

# ╔═╡ 1e35c150-29de-11eb-0069-cb75699b6d61
histogram(y[test],label="Label")

# ╔═╡ 76b4c570-29b5-11eb-32d5-75da0001a894
md"### Parâmetros encontrados"

# ╔═╡ 46a9fc40-2a8e-11eb-301f-077f9a6d50fe
md" Muitas vezes é interessante analisarmos a relevância dos coeficientes por váriosmotivos. Isto porque possibilita descobrir quais são as variáveis que ajudam mais num problema, ajudando inclusive na tomada de decisão de um ser humano. Adicionalmente, pode ser usado para compactar o modelo (ambientes com poucos recursos), eliminando atributos pouco relevantes. Finalmente, pode-se utilizar para verificar se seu modelo usa alguma variável que seja discrimnatória em termos de sociedade (ex. idade para tomada de crédito). "

# ╔═╡ 8d50cc90-294f-11eb-012b-3b973cbd1f1e
begin
	fp = fitted_params(pred_machine)
	coefs = fp.fitted_params_given_machine[fp.machines[2]]
	#fp.machines
end

# ╔═╡ 2d7233c0-29c4-11eb-0e0f-2f9b5773a062
begin
	names = [string(e[1]) for e in coefs[1] ]
	vals  = [e[2] for e in coefs[1] ]
	plot(names,vals,linetype = :bar,label="Coefs")
end	

# ╔═╡ dd923000-2a8e-11eb-1100-11a1ff5510bc
md"Coeficientes positivos significam que contribuem positivamente para o incremento de y. Os negativos, de forma oposta. Já os com valores próximos de zero, são pouco relevantes ao probelma."

# ╔═╡ eaad8980-29ab-11eb-059a-4fe1c6b0bb2d
md"### Análise de Erro"

# ╔═╡ 7cc2ab20-2a8d-11eb-2a73-e384f6d3b477
md" Separei aqui duas métricas que já vimos no curso: mae e rmse. Com o mae, conseguimos ter uma ideia do quanto erramos para cima ou para baixo com relação ao $y$. Erramos $3.2$ em termo de dinheiro para cima e para baixo do gabarito. Já o rmse indica uma variabilidade de $4.4$. Note que podemos ter um mae baixo, porém, se tivermos muito outliers, o rmse aumenta."

# ╔═╡ 2dadd5c0-2a8b-11eb-077d-d174e5c3e4dc
md"$\frac{1}{n}\sum_{i=1}^{n}{|ŷ - y|}$"

# ╔═╡ 1b9e5290-29a7-11eb-1c8f-4f12216b4391
md"mae = $(round(mae(ŷ, y[test]), sigdigits=4))"

# ╔═╡ a0b1a6f0-2a8b-11eb-1c89-d534b1671eec
md"$\sqrt{\frac{1}{n}\sum_{i=1}^{n}{(ŷ - y)^2}}$"

# ╔═╡ c3fb2bc0-295c-11eb-0061-c3875f5cb01b
md"rmse = $(round(rms(ŷ, y[test]), sigdigits=4))"
	

# ╔═╡ 0ebfd9e0-29ac-11eb-055f-615e53896367
md" Abaixo uma análise muito utilizada quando há uma relação de ordem temporal entre os exemplos. Apesar de neste problema não termos uma ordem temporal, deixo aqui este tipo de análise.Neste gráfico podemos notar que existem certos exemplos muito difíceis que talvez seja interessante analisar no futuro olhando seus atributos."

# ╔═╡ e102cfc0-295c-11eb-2c5b-834000ca2077
begin
	res = ŷ .- y[test]
	plot(res)
	scatter!(res)
end	

# ╔═╡ 2b478eb0-2a8c-11eb-0fe4-17dad212a408
md"Também é interessante analisarmos a distribuição do erro a partir de um histograma. Desta forma, temos uma noção maior da magnitude do erro e sinal. Da mesma forma que na análise anterior, vemos que existem exemplos que são bem mais difíceis que o normal. Cabe uma análise individual dos exemplos."

# ╔═╡ 845c9050-29aa-11eb-3587-1ff2b98c8905
histogram(res,label="Erro")

# ╔═╡ Cell order:
# ╟─ff5bd480-29c0-11eb-3112-cb35c17acc60
# ╠═61dc86f0-238f-11eb-054f-41a45b78e94d
# ╟─8e226310-29c0-11eb-1093-f95e53c029b6
# ╟─114c91a0-2a90-11eb-365b-4d360bb51ebe
# ╠═b6faf1b0-294a-11eb-38ae-0924c4943d81
# ╟─a2efe1c0-29c8-11eb-287d-49b90e597876
# ╟─a04c5f12-29d3-11eb-12a6-3f1b8acc88dc
# ╟─a417e700-29c8-11eb-3ee5-6d2c302d95ac
# ╟─9a810b20-29c0-11eb-1ca3-d155108c6e52
# ╟─91187080-2a8f-11eb-3001-1973d885e3f0
# ╠═2033e842-294f-11eb-2b38-1969d7440b88
# ╠═2f06c4a0-294f-11eb-2cb1-2da452f98b0f
# ╟─d2a9dfe0-29c0-11eb-36a9-e7e3c1d17717
# ╟─49b99b50-2a90-11eb-2b54-2704b914334d
# ╠═47a80140-294f-11eb-2880-ef7cfb0a8997
# ╟─e3d83ea0-29a3-11eb-10f5-b5f94102e1f1
# ╟─60083560-2a90-11eb-373f-3b8a1ba538a0
# ╠═868b73b0-29a9-11eb-0ef8-971446a3c919
# ╟─87cedb30-2a90-11eb-26dd-6162d49edd88
# ╠═6b75f5a0-294f-11eb-0835-cfe4a412cffe
# ╟─9abff260-2a90-11eb-026f-59a85ec6b64c
# ╠═fcde2920-29a6-11eb-37f7-4befad3ee3d2
# ╟─323d2620-29c0-11eb-258a-4703b4f5a0d2
# ╟─e57688a2-2a90-11eb-2272-3d9898bf4375
# ╠═fca39122-29a6-11eb-3b25-c7f0df7fc68f
# ╟─287ae7e0-2a91-11eb-0d5b-b138443bdc29
# ╠═fc474050-29a6-11eb-1fbd-87a88d38525a
# ╟─4449f8d0-29b5-11eb-3f83-c352d034ecb2
# ╟─91918630-2a91-11eb-1113-43b18086f208
# ╠═0de5e6e0-29a7-11eb-0558-2bb10a5fe1a5
# ╟─50f36580-29b5-11eb-06f2-cf4b9e142979
# ╟─ef445730-2a91-11eb-1552-c3e7d6e5ff34
# ╠═152f7100-29a7-11eb-2300-735ca72b15c3
# ╠═c29c1380-29b5-11eb-254c-2d026e15f903
# ╠═89d06380-29bf-11eb-0d5b-4533c94745d5
# ╠═8b3ac8f0-29bf-11eb-274e-f7b07eb35e7c
# ╠═1e35c150-29de-11eb-0069-cb75699b6d61
# ╟─76b4c570-29b5-11eb-32d5-75da0001a894
# ╟─46a9fc40-2a8e-11eb-301f-077f9a6d50fe
# ╟─8d50cc90-294f-11eb-012b-3b973cbd1f1e
# ╟─2d7233c0-29c4-11eb-0e0f-2f9b5773a062
# ╟─dd923000-2a8e-11eb-1100-11a1ff5510bc
# ╟─eaad8980-29ab-11eb-059a-4fe1c6b0bb2d
# ╟─7cc2ab20-2a8d-11eb-2a73-e384f6d3b477
# ╟─2dadd5c0-2a8b-11eb-077d-d174e5c3e4dc
# ╟─1b9e5290-29a7-11eb-1c8f-4f12216b4391
# ╟─a0b1a6f0-2a8b-11eb-1c89-d534b1671eec
# ╟─c3fb2bc0-295c-11eb-0061-c3875f5cb01b
# ╟─0ebfd9e0-29ac-11eb-055f-615e53896367
# ╟─e102cfc0-295c-11eb-2c5b-834000ca2077
# ╟─2b478eb0-2a8c-11eb-0fe4-17dad212a408
# ╟─845c9050-29aa-11eb-3587-1ff2b98c8905
