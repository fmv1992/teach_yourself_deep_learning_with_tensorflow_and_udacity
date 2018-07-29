# Lesson Thoughts

It wasn't clear for me how the word vectors would work. From what I could gather the word itself would be defined by the neighbouring words like this:

                      |-> word being defined by 4 words surrouding it
    word1   word2   word3   word4   word5

As an sparse array with shape (n_different_words, ) where it is 1 for the words that surround it and 0 for all other considered words.

# Problem 1
