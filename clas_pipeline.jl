### A Pluto.jl notebook ###
# v0.12.11

using Markdown
using InteractiveUtils

# ╔═╡ 61dc86f0-238f-11eb-054f-41a45b78e94d
begin
	using  MLJ
	using  MLJLinearModels
	using  CSV
	import RDatasets: dataset
    import DataFrames: describe, select, Not, rename!,head,DataFrame, nename!
	import Plots: plot,plot!,scatter,scatter!,histogram,histogram!
end	

# ╔═╡ ff5bd480-29c0-11eb-3112-cb35c17acc60
md" ### Libs"

# ╔═╡ a2efe1c0-29c8-11eb-287d-49b90e597876
md"### Sobre o Problema

url: https://archive.ics.uci.edu/ml/datasets/banknote+authentication

Data Set Information:

Data were extracted from images that were taken from genuine and forged banknote-like specimens. For digitization, an industrial camera usually used for print inspection was used. The final images have 400x 400 pixels. Due to the object lens and distance to the investigated object gray-scale pictures with a resolution of about 660 dpi were gained. Wavelet Transform tool were used to extract features from images.


Attribute Information:

1. variance of Wavelet Transformed image (continuous)
2. skewness of Wavelet Transformed image (continuous)
3. curtosis of Wavelet Transformed image (continuous)
4. entropy of image (continuous)
5. class (integer)
"

# ╔═╡ 8e226310-29c0-11eb-1093-f95e53c029b6
md"### Carga de Dados"

# ╔═╡ 114c91a0-2a90-11eb-365b-4d360bb51ebe
md" É interessante sempre mostrarmos os primeiros exemplos dos dados para ter uma ideia dos valores. Entretanto, como visto no curso, há várias análises possíveis para se fazer como identificar nulos, outliers, ... Tudo isto, é bom ser tratado."

# ╔═╡ 1c623cf0-2d17-11eb-2f20-ad2b49b221de
begin
	url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
	tmp = download(url)
end	

# ╔═╡ 049db9d0-2d19-11eb-24f1-0ba111942cb8
begin 
	banknote = DataFrame(CSV.File(tmp; header=false))
	rename!(banknote, Dict(:Column1 => "Variance", :Column2 => "Skewness",:Column3 => "Curtosis",:Column4 => "Entropy",:Column5 => "Target")) 
end	

# ╔═╡ 9a810b20-29c0-11eb-1ca3-d155108c6e52
md"### Estatísticas dos Dados"

# ╔═╡ 91187080-2a8f-11eb-3001-1973d885e3f0
md"É sempre bom tiramos umas estatísticas dos dados para entender melhor o comportamento das variáveis do nosso problema. Qual a variância de cada variável? Elas estão na mesma ordem de grandeza? Máixmo e minímo ?"

# ╔═╡ 2033e842-294f-11eb-2b38-1969d7440b88
describe(banknote, :mean, :std, :min, :max, :eltype)

# ╔═╡ 2f06c4a0-294f-11eb-2cb1-2da452f98b0f
begin
	data = coerce(banknote, autotype(banknote, :discrete_to_continuous));
	data = coerce(data,:Target => OrderedFactor)#Multiclass)
end

# ╔═╡ d2a9dfe0-29c0-11eb-36a9-e7e3c1d17717
md"### Selecionando Dados"

# ╔═╡ 49b99b50-2a90-11eb-2b54-2704b914334d
md"É nesta fase que você vai escolher seu target e vai selecionar os atributos para trabalhar."

# ╔═╡ 47a80140-294f-11eb-2880-ef7cfb0a8997
begin
	y = data.Target;
    X = select(data, Not(:Target));
end	

# ╔═╡ 429fc960-2d25-11eb-3c06-614daa2fc78f
schema(X)

# ╔═╡ e3d83ea0-29a3-11eb-10f5-b5f94102e1f1
md"### Verificando Algoritmos Disponíveis"

# ╔═╡ 60083560-2a90-11eb-373f-3b8a1ba538a0
md"Essa é uma característica bem legal do MLJ. Como nosso dados possuem atributos numéricos e o target numérico, ele me mostra todos os algoritmos que conseguem rodar para este tipo de dado."

# ╔═╡ 868b73b0-29a9-11eb-0ef8-971446a3c919
#models(matching(X, y))
begin
	task(model) = matching(model, X, y) && model.prediction_type == :probabilistic
	models(task)
end

# ╔═╡ 87cedb30-2a90-11eb-26dd-6162d49edd88
md"Abaixo realizamos uma carga no nosso regressor linear."

# ╔═╡ 9abff260-2a90-11eb-026f-59a85ec6b64c
md"O MLJ nos permite montar um pipeline. Pipeline é uma forma de montar uma sequência de operações poupando-lhe código. No nosso caso, queremos padronizar X e y antes de realizarmos o treino do regressor. Podemos fazer isso em uma única linha."

# ╔═╡ fcde2920-29a6-11eb-37f7-4befad3ee3d2
predictor = @pipeline  Standardizer LogisticClassifier(lambda = 0.5)

# ╔═╡ 323d2620-29c0-11eb-258a-4703b4f5a0d2
md"### Partição dos Dados"

# ╔═╡ e57688a2-2a90-11eb-2272-3d9898bf4375
md"Realizamos aqui o que conhecemos como hold-out. Particionamos em treino e teste, que é a forma mais simples de se fazer uma validação. Sabemos que existem outras no curso. No caso, deixei 70% para treino e 30% para teste. Como podem notar, a função gera índices qu correspondem às linhas dos exemplos. Ele não tira cópia dos dados."

# ╔═╡ fca39122-29a6-11eb-3b25-c7f0df7fc68f
train, test  = partition(eachindex(y), 0.7, shuffle=true);

# ╔═╡ 287ae7e0-2a91-11eb-0d5b-b138443bdc29
md"Um conceito interessante do MLJ é o machine. Esta função faz uma mapeamento do preditor ao dado. Quando aplicamos machine, estamos gerando um nó num grafo de computação e um estado associado. Existem algmas vantagens neste paradigma como: associar um mesmo preditor a vários dados distintos que podem ser processados em diferentes recursos; evitar re execução de um mesmo preditor e um mesmo dado."

# ╔═╡ fc474050-29a6-11eb-1fbd-87a88d38525a
pred_machine = machine(predictor, X, y);

# ╔═╡ 4449f8d0-29b5-11eb-3f83-c352d034ecb2
md"### Treino do Classificador"

# ╔═╡ 91918630-2a91-11eb-1113-43b18086f208
md" Aqui temos enfim, o treino do nosso regressor para o data X, apenas no conjunto de treino. Isto porque estamos passando os índices do exemplos."

# ╔═╡ 03caa2a0-2d2a-11eb-0742-413f150b9056
fit!(pred_machine, rows=train);

# ╔═╡ 50f36580-29b5-11eb-06f2-cf4b9e142979
md"### Análise da Predição"

# ╔═╡ ef445730-2a91-11eb-1552-c3e7d6e5ff34
md"Aqui analisamos cada predição e cada target. Este tipo de análise é mais apropriada quando nosso conjunto de dados possui uma relação de ordem temporal entre os exemplos (que não é o caso). Entretanto, é uma análise boa a se fazer. Podemos notar que há um atendência de subestimar os valores mais altos do que valores mais baixos. Você pode analisar o que pode estar gerando dificldade nisto analisando alguns exemplos individualmente."

# ╔═╡ 36bd3310-2dbd-11eb-0e46-bf6c0b1e80a9
md"Aqui pegamos as classes e probabilidades"

# ╔═╡ 152f7100-29a7-11eb-2300-735ca72b15c3
preds = predict(pred_machine, rows=test);

# ╔═╡ 275bab42-2dbd-11eb-1bd7-a1c0cb2965a7
md"Pegando probabilidades da predição por classe" 

# ╔═╡ 1336f910-2d9c-11eb-222b-ade11f5b7b51
begin
	probs_fraude  = broadcast(pdf,preds,1.0);
	probs_nfraude = broadcast(pdf,preds,0.0);
end	

# ╔═╡ d4536cb0-2d9b-11eb-16f7-993d0065c683
#mode.(ŷ)
ŷ = predict_mode(pred_machine,rows = test);

# ╔═╡ 35b079a0-2d9f-11eb-0fa6-1bb075367008
measures(matching(y))

# ╔═╡ dbb70eb2-2dc0-11eb-3da4-754b2e0de392
md"#### Métricas para o limiar padrão (0.5)"

# ╔═╡ 32c66e80-2d9e-11eb-220e-093e86fdc72f
cm = confusion_matrix(ŷ,y[test])

# ╔═╡ 3593e090-2e05-11eb-0557-13c977ff89b2
false_positive_rate(cm),false_negative_rate(cm)

# ╔═╡ 463a4470-2e05-11eb-0cac-2f56d62bfa20
true_positive_rate(cm),true_negative_rate(cm) # recall

# ╔═╡ 1fedd0a0-2e07-11eb-1f26-8575ef999663
positive_predictive_value(cm),negative_predictive_value(cm) # precision

# ╔═╡ 6a6e0550-2d9e-11eb-1136-119a2997dcae
accuracy(cm)

# ╔═╡ 0af080a0-2e05-11eb-0069-8d908435f94b
balanced_accuracy(ŷ,y[test])

# ╔═╡ c8288560-2dc8-11eb-14f6-19ae67fb2357
begin
	f1 = FScore{1.0}()
	f1(cm)
end	

# ╔═╡ 00989c50-2dc9-11eb-31fc-0d8ab39b5a92
matthews_correlation(cm)

# ╔═╡ 77646ed0-2dc5-11eb-35c5-4b1582878184
md"#### Métricas para limiares distintos"

# ╔═╡ f3292f10-2d9d-11eb-1e76-8de80adce2e6
tprs, fprs, ts = roc_curve(preds,y[test]);

# ╔═╡ 96c67790-2e07-11eb-0000-19536e55469e
ts

# ╔═╡ 903e2090-2dc5-11eb-2855-bfc16c6145b6
md" Área embaixo da curva"

# ╔═╡ 58f2c540-2d9e-11eb-3a14-0931bcd2cc5d
auc(preds,y[test])

# ╔═╡ c29c1380-29b5-11eb-254c-2d026e15f903
plot(tprs,fprs,label="LG Predictions",ylabel="Verdadeiros Positivos",xlabel="Falsos Positivos")

# ╔═╡ 89d06380-29bf-11eb-0d5b-4533c94745d5
md"#### Distribuição das probabilidades das predições"

# ╔═╡ 8b3ac8f0-29bf-11eb-274e-f7b07eb35e7c
histogram(probs_fraude,bins=10,label="Fraude")

# ╔═╡ 1e35c150-29de-11eb-0069-cb75699b6d61
histogram(probs_nfraude,bins=10,label="Não Fraude")

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

# ╔═╡ e102cfc0-295c-11eb-2c5b-834000ca2077
#begin
#	res = ŷ .- y[test]
#	plot(res)
#	scatter!(res)
#end	

# ╔═╡ 08b31090-2e01-11eb-27a7-c3758492386e
begin
	names = [string(e[1]) for e in coefs[2] ]
	vals  = [e[2] for e in coefs[2] ]
	plot(names,vals,linetype = :bar,label="Coefs")
end	

# ╔═╡ Cell order:
# ╟─ff5bd480-29c0-11eb-3112-cb35c17acc60
# ╠═61dc86f0-238f-11eb-054f-41a45b78e94d
# ╟─a2efe1c0-29c8-11eb-287d-49b90e597876
# ╟─8e226310-29c0-11eb-1093-f95e53c029b6
# ╟─114c91a0-2a90-11eb-365b-4d360bb51ebe
# ╠═1c623cf0-2d17-11eb-2f20-ad2b49b221de
# ╠═049db9d0-2d19-11eb-24f1-0ba111942cb8
# ╟─9a810b20-29c0-11eb-1ca3-d155108c6e52
# ╟─91187080-2a8f-11eb-3001-1973d885e3f0
# ╠═2033e842-294f-11eb-2b38-1969d7440b88
# ╠═2f06c4a0-294f-11eb-2cb1-2da452f98b0f
# ╟─d2a9dfe0-29c0-11eb-36a9-e7e3c1d17717
# ╟─49b99b50-2a90-11eb-2b54-2704b914334d
# ╠═47a80140-294f-11eb-2880-ef7cfb0a8997
# ╠═429fc960-2d25-11eb-3c06-614daa2fc78f
# ╟─e3d83ea0-29a3-11eb-10f5-b5f94102e1f1
# ╟─60083560-2a90-11eb-373f-3b8a1ba538a0
# ╠═868b73b0-29a9-11eb-0ef8-971446a3c919
# ╟─87cedb30-2a90-11eb-26dd-6162d49edd88
# ╟─9abff260-2a90-11eb-026f-59a85ec6b64c
# ╠═fcde2920-29a6-11eb-37f7-4befad3ee3d2
# ╟─323d2620-29c0-11eb-258a-4703b4f5a0d2
# ╟─e57688a2-2a90-11eb-2272-3d9898bf4375
# ╠═fca39122-29a6-11eb-3b25-c7f0df7fc68f
# ╟─287ae7e0-2a91-11eb-0d5b-b138443bdc29
# ╠═fc474050-29a6-11eb-1fbd-87a88d38525a
# ╟─4449f8d0-29b5-11eb-3f83-c352d034ecb2
# ╟─91918630-2a91-11eb-1113-43b18086f208
# ╠═03caa2a0-2d2a-11eb-0742-413f150b9056
# ╟─50f36580-29b5-11eb-06f2-cf4b9e142979
# ╟─ef445730-2a91-11eb-1552-c3e7d6e5ff34
# ╟─36bd3310-2dbd-11eb-0e46-bf6c0b1e80a9
# ╠═152f7100-29a7-11eb-2300-735ca72b15c3
# ╟─275bab42-2dbd-11eb-1bd7-a1c0cb2965a7
# ╠═1336f910-2d9c-11eb-222b-ade11f5b7b51
# ╠═d4536cb0-2d9b-11eb-16f7-993d0065c683
# ╠═35b079a0-2d9f-11eb-0fa6-1bb075367008
# ╟─dbb70eb2-2dc0-11eb-3da4-754b2e0de392
# ╠═32c66e80-2d9e-11eb-220e-093e86fdc72f
# ╠═3593e090-2e05-11eb-0557-13c977ff89b2
# ╠═463a4470-2e05-11eb-0cac-2f56d62bfa20
# ╠═1fedd0a0-2e07-11eb-1f26-8575ef999663
# ╠═6a6e0550-2d9e-11eb-1136-119a2997dcae
# ╠═0af080a0-2e05-11eb-0069-8d908435f94b
# ╠═c8288560-2dc8-11eb-14f6-19ae67fb2357
# ╠═00989c50-2dc9-11eb-31fc-0d8ab39b5a92
# ╠═77646ed0-2dc5-11eb-35c5-4b1582878184
# ╠═f3292f10-2d9d-11eb-1e76-8de80adce2e6
# ╠═96c67790-2e07-11eb-0000-19536e55469e
# ╟─903e2090-2dc5-11eb-2855-bfc16c6145b6
# ╠═58f2c540-2d9e-11eb-3a14-0931bcd2cc5d
# ╠═c29c1380-29b5-11eb-254c-2d026e15f903
# ╟─89d06380-29bf-11eb-0d5b-4533c94745d5
# ╟─8b3ac8f0-29bf-11eb-274e-f7b07eb35e7c
# ╟─1e35c150-29de-11eb-0069-cb75699b6d61
# ╟─76b4c570-29b5-11eb-32d5-75da0001a894
# ╟─46a9fc40-2a8e-11eb-301f-077f9a6d50fe
# ╠═8d50cc90-294f-11eb-012b-3b973cbd1f1e
# ╟─e102cfc0-295c-11eb-2c5b-834000ca2077
# ╟─08b31090-2e01-11eb-27a7-c3758492386e
