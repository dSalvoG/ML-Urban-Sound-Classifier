import functions
from functions import print_prediction


class_names = ['Air Conditioner', 'Car Horn', 'Children Playing', 'Dog Bark', 
               'Drilling', 'Engine Idling', 'Gun Shot', 
               'Jackhammer', 'Siren', 'Street Music']

# Loading Features and Label arrays
features_test = np.load('features_test5.npy')
labels_test = np.load('labels_test5.npy')

test_loss, test_acc = model.evaluate(features_test, labels_test, verbose=2)

# Prediction for Dog Bark example FOLD 5
parent_dir = 'train'
sub_dirs= ['dog']
print_prediction(parent_dir,sub_dirs)