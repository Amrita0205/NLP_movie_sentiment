import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

import matplotlib.pyplot as plt

sentences = [
    'I loved the movie',
    'This movie was amazing',
    'This was a fantastic film',
    'Amazing acting and story',
    'I really enjoyed watching this',
    'Best movie I have seen this year',
    'Absolutely wonderful experience',
    'The film was brilliant and touching',
    'Great direction and performances',
    'I would watch this again',
    'A delightful and heartwarming movie',
    'I hated this movie',
    'This was a terrible film',
    'Bad acting and boring story',
    'I really disliked watching this',
    'Worst movie I have seen this year',
    'Absolutely awful experience',
    'The film was dull and disappointing',
    'Poor direction and performances',
    'I would never watch this again',
    'A boring and forgettable movie'
]

labels = [1] * 11 + [0] * 10  # This just means that it's like [1,1,1...1,0,0,0,...0]

tfidf = TfidfVectorizer()
vectorized_sentences = tfidf.fit_transform(sentences).toarray()
# print(tfidf)
print(vectorized_sentences.shape)  # returns the size as tuple with indexes 0 , 1
# print(vectorized_sentences[:2])

vocab_size = vectorized_sentences.shape[1]
print(len(labels))  # 21 clearly this is just 1-D we need a tensor of (21,1)

# converting X,y so that the model accepts it
X_tensor = torch.tensor(vectorized_sentences, dtype=torch.float32)
y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

print(X_tensor.shape, y_tensor.shape)


class SentimentClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layer1 = nn.Linear(
            input_size, hidden_size
        )  # here hidden_size is the hyper parameter that we decided
        self.layer2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x


# hidden_sizes_to_test=[2,8,16,32,64]
hidden_sizes_to_test = [2**i for i in range(8)]   # [1, 2, 4, 8, 16, 32, 64, 128]
results={}
epochs=200

for hidden_size in hidden_sizes_to_test:
    torch.manual_seed(42)# SAME seed each time -> fair comparison for the initial weights
    model = SentimentClassifier(input_size=vocab_size,hidden_size=hidden_size) 
    # predictions = model(X_tensor)
    criterion = nn.BCELoss() #deciding which loss to use
    # loss = criterion(predictions, y_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    loss_history = []
    
    for epoch in range(epochs):
        predictions = model(X_tensor)
        loss = criterion(predictions, y_tensor) #calculating the loss values
        optimizer.zero_grad() #flushing gradient values per epoch
        loss.backward() #applies back_propagation using gradient descent
        optimizer.step() #this is updating the gradients during back propagation
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
            
        loss_history.append(loss.item())
    results[hidden_size] = loss_history
    # print(f"Hidden size {hidden_size:3d} -> final loss: {loss_history[-1]:.5f}")        
            
# ---- Plot ----
plt.figure(figsize=(9, 6))
for hidden_size, loss_history in results.items():
    plt.plot(loss_history, label=f"hidden_size={hidden_size}")

plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.title("Effect of Hidden Layer Size on Training Loss")
plt.legend()
plt.yscale("log")
plt.grid(alpha=0.3)
plt.show()
            
            
test_sentences = [
    'I hated the movie',
    'The movie was the best in my life',
    'this movie was an utter waste of money',
    'Im never recommending this to anyone',
    'Id encourage people to watch this'
]

X_test = tfidf.transform(test_sentences).toarray()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    test_predictions = model(X_test_tensor)
print(test_predictions)
