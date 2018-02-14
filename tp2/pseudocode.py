pour chaque epoque :
    # forward pass
    pour chaque hidden layer l
        # h est la matrice des fonctions d'activation d'un layer
        # h(0) = features du réseau
        # a = theta(l) * h(l-1)
        # h(i, n+1) = 1 pour tout i, n = nb features car on impose un bias dans le réseau
        a(l) = theta(l) * h(l-1)
        h(l) = sigmoid(- a)
    a(l+1) = theta(L+1) * h(L)
    output = sigmoid(- a(l+1))
    # L = nb hidden layers

    # error compute
    delta(L+1) = (y - output)^2

    # backward pass
    pour chaque hidden layer l dans l'ordre inverse
        # compute the derivative of Wij and update it (with lr)
        d_error = -(y - output)
        d_sigm = h(l) * (1 - h(l))
        delta(l) = d_sigm * theta(l+1).T * delta(l+1)
        theta(l) += lr * - delta(l) * h(l-1).T

Question b)
J'utiliserai la technique du SGD vue dans le TP1 (en faisant la batch normalization).
Le pseudocode aurait besoin de quelques ajustements pour
tenir compte de la taille du batch dans les calculs.
De même, il serait souhaitable de lui ajouter une regularisation.
Il serait intéressant d'utiliser des tailles de batch approchant la capacité de
parallélisme du GPU utilisé.
Quand au learning rate, il faudrait tester soit par grid search, soit par decay.
Il serait peut-être intéressant de commencer avec un nombre de hidden layers limités
pour tester ces 2 paramètres, puis à augmenter la profondeur du réseau pour affiner l'accuracy.
Enfin, pour limiter l'effet de surapprentissage il faudrait limiter la durée/le nombre d'epoch.
