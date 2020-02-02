import numpy as np
import re
import nltk
nltk.corpus.gutenberg.fileids()
['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt',
'blake-poems.txt', 'bryant-stories.txt', 'burgess-busterbrown.txt',
'carroll-alice.txt', 'chesterton-ball.txt', 'chesterton-brown.txt',
'chesterton-thursday.txt', 'edgeworth-parents.txt', 'melville-moby_dick.txt',
'milton-paradise.txt', 'shakespeare-caesar.txt', 'shakespeare-hamlet.txt',
'shakespeare-macbeth.txt', 'whitman-leaves.txt']

book1 = nltk.corpus.gutenberg.words('austen-emma.txt')
book2 = nltk.corpus.gutenberg.words('bible-kjv.txt')
book3 = nltk.corpus.gutenberg.words('chesterton-ball.txt')
book4 = nltk.corpus.gutenberg.words('shakespeare-hamlet.txt')
book5 = nltk.corpus.gutenberg.words('blake-poems.txt')
book6 = nltk.corpus.gutenberg.words('bryant-stories.txt')
book7 = nltk.corpus.gutenberg.words('whitman-leaves.txt')



#books = np.array (['austen-emma.txt', 'bilble-kjv.txt', 'chesterton-ball.txt', 'whitman-leaves.txt',
#'blake-poems.txt', 'bryant-stories.txt'])



from nltk.stem import WordNetLemmatizer


for sen in range(0, len(book1)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(book1[sen]))
    
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Converting to Lowercase
    document = document.lower()
    
    # Lemmatization
    document = document.split()


    #document = [stemmer.lemmatize(word) for word in document]
   # document = ' '.join(document)
   # documents.append(document)

print('result')
print(document)