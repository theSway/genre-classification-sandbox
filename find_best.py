from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


def support_vector_classifier(x_train, y_train, x_test, y_test):
    # Support vector classifier
    kernel_params = ['linear', 'poly', 'rbf', 'sigmoid']

    current_best_score = 0
    current_best_svc = None
    for kernel in kernel_params:
        clf = SVC(kernel=kernel, gamma='auto', probability=True)
        svc_fit = clf.fit(x_train, y_train)
        svc_score = svc_fit.score(x_test, y_test)
        if svc_score > current_best_score:
            current_best_score = svc_score
            current_best_svc = svc_fit


    return current_best_svc, current_best_score

def decision_tree_classifier(x_train, y_train, x_test, y_test):
    criterion_params = ['gini', 'entropy']
    class_weight_params = [None, 'balanced']

    current_best_score = 0
    current_best_tree = None
    for criterion in criterion_params:
        for class_weight in class_weight_params:
            clf = tree.DecisionTreeClassifier(criterion=criterion, class_weight=class_weight)
            tree_fit = clf.fit(x_train, y_train)
            tree_score = tree_fit.score(x_test, y_test)
            if tree_score > current_best_score:
                current_best_score = tree_score
                current_best_tree = tree_fit

    return current_best_tree, current_best_score

def naive_bayes(x_train, y_train, x_test, y_test):
    prior_params = [None, [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]  # Balanced and uniform

    current_best_score = 0
    current_best_bayes = None
    for prior in prior_params:
        clf = GaussianNB(priors=prior)
        bayesian_fit = clf.fit(x_train, y_train)
        bayesian_score = bayesian_fit.score(x_test, y_test)
        if bayesian_score > current_best_score:
            current_best_score = bayesian_score
            current_best_bayes = bayesian_fit

    return current_best_bayes, current_best_score

def random_forest(x_train, y_train, x_test, y_test):

    n_estimator_params = [100, 500]
    criterion_params = ['gini', 'entropy']
    warm_start_params = [False, True]

    learning_rate_params = [0.05, 0.1]

    current_best_score = 0
    current_best_forest = None
    for estimator in n_estimator_params:
        for criterion in criterion_params:
            for warm_start in warm_start_params:
                clf = ExtraTreesClassifier(n_jobs=-1,
                                           n_estimators=estimator,
                                           criterion=criterion,
                                           warm_start=warm_start)
                forest_fit = clf.fit(x_train, y_train)
                forest_score = forest_fit.score(x_test, y_test)
                if forest_score > current_best_score:
                    current_best_score = forest_score
                    current_best_forest = forest_fit

    for estimator in n_estimator_params:
        for criterion in criterion_params:
            for warm_start in warm_start_params:
                clf = RandomForestClassifier(n_jobs=-1,
                                             n_estimators=estimator,
                                             criterion=criterion,
                                             warm_start=warm_start)
                forest_fit = clf.fit(x_train, y_train)
                forest_score = forest_fit.score(x_test, y_test)
                if forest_score > current_best_score:
                    current_best_score = forest_score
                    current_best_forest = forest_fit

    for estimator in n_estimator_params:
        for warm_start in warm_start_params:
            for rate in learning_rate_params:
                clf = GradientBoostingClassifier(n_estimators=estimator,
                                                 warm_start=warm_start,
                                                 learning_rate=rate)
                forest_fit = clf.fit(x_train, y_train)
                forest_score = forest_fit.score(x_test, y_test)
                if forest_score > current_best_score:
                    current_best_score = forest_score
                    current_best_forest = forest_fit

    return current_best_forest, current_best_score

def knn_classifier(data, x_train, y_train, x_test, y_test, n_components=35):
    # Performs on PCA

    pca = PCA(n_components=n_components).fit(data.drop(labels=0, axis='columns'))
    data_train_reduced = pca.transform(x_train)
    data_test_reduced = pca.transform(x_test)

    # knn
    neighbor_params = [4, 5, 6, 7, 8, 9, 10]
    weights_params = ['uniform', 'distance']

    current_best_score = 0
    current_best_knn = None
    for neighbors in neighbor_params:
        for weight in weights_params:
            knn_classifier = knn(n_jobs=-1, n_neighbors=neighbors, weights=weight)
            knn_fit = knn_classifier.fit(data_train_reduced, y_train)
            knn_score = knn_fit.score(data_test_reduced, y_test)
            if knn_score > current_best_score:
                current_best_score = knn_score
                current_best_knn = knn_fit

    return current_best_knn, current_best_score


def neural_network(x_train, y_train, x_test, y_test):
    hidden_layer_size_params = [10, 100]
    activation_params = ['identity', 'logistic', 'tanh', 'relu']
    warm_start_params = [False, True]

    current_best_score = 0
    current_best_nn = None
    for size in hidden_layer_size_params:
        for activation in activation_params:
            for warm_start in warm_start_params:
                clf = MLPClassifier(hidden_layer_sizes=(size,),
                                    activation=activation,
                                    warm_start=warm_start,
                                    max_iter=300)
                nn_fit = clf.fit(x_train, y_train)
                nn_score = nn_fit.score(x_test, y_test)
                if nn_score > current_best_score:
                    current_best_score = nn_score
                    current_best_nn = nn_fit

    return current_best_nn, current_best_score

