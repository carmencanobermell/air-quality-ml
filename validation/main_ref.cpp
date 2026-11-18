// Copy of the original main.cpp without windows.h: CSV path from argv, results dumped to validate the Python version.
#include <cmath>
#include <iostream>
#include <map>
#include <algorithm>
#include "data_loader.h"
#include "clustering.h"
#include "stats.h"
#include "rules.h"
#include "model.h"

int main(int argc, char** argv) {
    vector<DataPoint> datos = cargarCSV(argv[1]);
    normalizar(datos);
    // K-Means: 10 runs, labels written to a file
    ofstream out(argv[2]);
    for (int r = 0; r < 10; ++r) {
        vector<int> et = kmeans(datos, 5);
        for (size_t i = 0; i < et.size(); ++i) out << et[i] << (i + 1 < et.size() ? "," : "\n");
    }
    out.close();
    analizarCorrelaciones(datos);
    generarReglasFrecuentes(datos);
    vector<double> valores;
    for (auto& d : datos) valores.push_back(d.features[2]);
    sort(valores.begin(), valores.end());
    double umbral = valores[valores.size() * 0.65];
    etiquetarPorUmbral(datos, umbral);
    vector<DataPoint> train, test;
    dividirDatos(datos, train, test);
    int positivos = count_if(datos.begin(), datos.end(), [](DataPoint d) { return d.label == 1; });
    cout << "Positivos: " << positivos << ", Negativos: " << datos.size() - positivos << "\n";
    cout << "Train: " << train.size() << " Test: " << test.size() << "\n";
    LogisticRegression modelo;
    modelo.entrenar(train);
    cout << "Pesos:";
    for (double w : modelo.pesos) cout << " " << w;
    cout << "\n";
    evaluarModelo(modelo, test);
    return 0;
}
