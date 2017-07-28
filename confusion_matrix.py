import pandas
from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt

fields = ['image_url', 'emotion']

#image_emotion_map = pandas.read_excel(io = "/Users/saila/Desktop/Environments/mturk_data_analysis/image_emotion_map.csv", sheetname="All_image_accuracies", index_col ="None")
image_emotion_map = pandas.read_csv("/Users/saila/Desktop/Environments/mturk_data_analysis/image_emotion_map.csv", usecols=fields)

#print(image_emotion_map.keys())

#print(image_emotion_map.image_url.head())
print(image_emotion_map.head())

emotion_dictionary = dict(zip(list(image_emotion_map.image_url), list(image_emotion_map.emotion)))
#print(emotion_dictionary.keys().head())


#Batch123- only approved HITS
#all_results - both approved and rejected HITs
results = pandas.read_csv("/Users/saila/Desktop/Environments/mturk_data_analysis/Batch_1_2_3_results.csv")
print(results.head())


results_mapped = results.replace({"image_url": emotion_dictionary})

print(results_mapped.head())

print(results_mapped.image_url.unique())

y_true = list(results_mapped.image_url)
y_crowdsourced = list(results_mapped.crowdemotion1)
print(len(y_true))
print(len(y_crowdsourced))

#print(y_true)
#print(y_crowdsourced)

confusion_matrix = ConfusionMatrix(y_true, y_crowdsourced)
print("Confusion matrix:\n%s" % confusion_matrix)


confusion_matrix.plot()
plt.show()
