"""Importe os pacotes necessários"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1, help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

"""Pegue a lista de imagens que estaremos descrevendo"""
print("[INFO] loading images...")
image_paths = list(paths.list_images(args['dataset']))

"""
Inicialize o pré-processador de imagens, carregue a base de dados a partir do disco
e remodele a matrix de dados
"""
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(image_paths, verbose=500)
data = data.reshape((data.shape[0], 3072))

"""Mostre alguma informação a respeito do consumo de memória das imagens"""
print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))

"""Codifique as labels como inteiros"""
le = LabelEncoder()
labels = le.fit_transform(labels)

"""
Particione os dados em divisões de treino e teste usando 75% dos dados para treino
e os restantes 25% para teste
"""
(X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size=0.25, random_state=42)

"""Treine e avalie um classificador k-NN em usar as intensidades puras dos pixeis"""
print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=args['neighbors'], n_jobs=args['jobs'])
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test), target_names=le.classes_))