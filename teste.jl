using Plots, StatsPlots,MLJ, Queryverse,
 Statistics, KernelDensity,PrettyPrinting
plotly()

dados = Queryverse.load("iris.csv") |> DataFrame
coerce!(dados, autotype(dados))
schema(dados)

y, x = unpack(dados, ==(:variety), colnames -> true)
for m in models(matching(x,y))
    if m.prediction_type == :probabilistic &&
        m.is_pure_julia == true
        println(rpad(m.name, 28), m.package_name)
    end
end

@load DecisionTreeClassifier pkg="DecisionTree"
dtc = DecisionTreeClassifier()

arvore = machine(dtc, x, y)

treino, teste = partition(eachindex(y), 0.7, shuffle = true);

fit!(arvore, rows=treino, force = true)

fitted_params(arvore)

report(arvore)

report(arvore).print_tree()

y_pred = predict(arvore, rows = teste)

y_pred_mode = predict_mode(arvore, rows = teste)

y_pred_mode |> pprint

for m in measures()
    println(rpad(m.name, 29), m.orientation)
end

for m in measures()
    if m.orientation == :score
        println(rpad(m.name, 29))
    end
end

accuracy(y_pred_mode, y[teste]) |> pprint
balanced_accuracy(y_pred_mode, y[teste]) |> pprint

confusion_matrix(y_pred_mode, y[teste])
