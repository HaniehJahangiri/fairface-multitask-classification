import matplotlib.pyplot as plt
#plotting losses
plt.figure(figsize=(16,8))

plt.subplot(2,2,1)
plt.plot(train_losses,label='Train Loss')
plt.plot(valid_losses,label='Test Loss')
plt.title('Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.tight_layout(pad= 3)
plt.legend()

#Plotting Gender Accuracies
plt.subplot(2,2,2)
plt.plot(train_gender_accuracies,label='Train Gender Accuracy')
plt.plot(valid_gender_accuracies,label='Test Gender Accuracy')
plt.title('Gender Accuracies')
plt.xlabel('Epochs')
plt.ylabel('Gender Accuracy')
plt.tight_layout(pad= 3)
plt.legend()

#Plotting Race Accuracies
plt.subplot(2,2,3)
plt.plot(train_race_accuracies,label='Train Race Accuracy')
plt.plot(valid_race_accuracies,label='Test Race Accuracy')
plt.title('Race Accuracies')
plt.xlabel('Epochs')
plt.ylabel('Race Accuracy')
plt.tight_layout(pad= 3)
plt.legend()

#Plotting Age Mean Absolute Error
plt.subplot(2,2,4)
plt.plot(train_age_maes,label='Train Age MAE')
plt.plot(valid_age_maes,label='Test Age MAE')
plt.title('Age MAEs')
plt.xlabel('Epochs')
plt.ylabel('Age MAE')
plt.tight_layout(pad= 3)
plt.legend()

plt.show()