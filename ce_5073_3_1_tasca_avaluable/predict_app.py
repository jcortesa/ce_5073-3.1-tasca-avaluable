import pickle
from flask import Flask, jsonify, request
from predict_service import (predict_logistic_regression, 
                            predict_support_vector_machine, 
                            predict_decision_tree,
                            predict_k_nearest_neighbours)

app = Flask('iris-predict')

labels = {
    0: 'Iris Setosa',
    1: 'Iris Versicolour',
    2: 'Iris Virginica'
}

with open('models/1-logistic-regression-model.pck', 'rb') as f:
    lr_sc, lr_model = pickle.load(f)

with open('models/2-support-vector-machine-model.pck', 'rb') as f:
    svm_sc, svm_model = pickle.load(f)

with open('models/3-decision-tree-model.pck', 'rb') as f:
    dt_model = pickle.load(f)

with open('models/4-k-nearest-neighbours-model.pck', 'rb') as f:
    knn_sc, knn_model = pickle.load(f)

@app.route('/logistic-regression/predict', methods=['POST'])
def logistic_regression_predict():
    flower_data = request.get_json()
    petal_length = float(flower_data['petal_length'])
    petal_width = float(flower_data['petal_width'])

    class_label, class_probability = predict_logistic_regression(petal_length,
                                                                 petal_width,
                                                                 lr_sc,
                                                                 lr_model)
    
    result = {
        'probable flower name': labels[class_label],
        'probability of match': '{0:.5g}%'.format(class_probability * 100),
    }

    return jsonify(result)

@app.route('/support-vector-machine/predict', methods=['POST'])
def support_vector_machine_predict():
    flower_data = request.get_json()
    petal_length = float(flower_data['petal_length'])
    petal_width = float(flower_data['petal_width'])

    class_label, class_probability = predict_support_vector_machine(petal_length,
                                                                 petal_width,
                                                                 lr_sc,
                                                                 lr_model)
    
    result = {
        'probable flower name': labels[class_label],
        'probability of match': '{0:.5g}%'.format(class_probability * 100),
    }

    return jsonify(result)


@app.route('/decision-tree/predict', methods=['POST'])
def decision_tree_predict():
    flower_data = request.get_json()
    petal_length = float(flower_data['petal_length'])
    petal_width = float(flower_data['petal_width'])

    class_label, class_probability = predict_decision_tree(petal_length,
                                                                 petal_width,
                                                                 dt_model)
    
    result = {
        'probable flower name': labels[class_label],
        'probability of match': '{0:.5g}%'.format(class_probability * 100),
    }

    return jsonify(result)

@app.route('/k-nearest-neighbours/predict', methods=['POST'])
def k_nearest_neighbours_predict():
    flower_data = request.get_json()
    petal_length = float(flower_data['petal_length'])
    petal_width = float(flower_data['petal_width'])

    class_label, class_probability = predict_k_nearest_neighbours(petal_length,
                                                                 petal_width,
                                                                 knn_sc,
                                                                 knn_model)
    
    result = {
        'probable flower name': labels[class_label],
        'probability of match': '{0:.5g}%'.format(class_probability * 100),
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=8000)