import matplotlib.pyplot as plt

methods = ['Keyword Filter', 'Proposed System']
accuracy = [55, 82]  # Accuracy percentages

plt.bar(methods, accuracy, color=['red', 'blue'])
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Comparison')
plt.ylim(0, 100)
plt.show()
