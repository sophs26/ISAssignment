load('wordembedding.mat');

%load the opinion lexicon
data = readLexicon;
%view some words that are labeled as positive then negative
idx= data.Label == "Positive";
head(data(idx,:))
idx = data.Label == "Negative";
head(data(idx,:))

%preparing the data for the training
idx = ~isVocabularyWord(emb,data.Word);
data(idx,:)=[];
%get the number of words
numWords = size(data,1);
%Split up words, some for training, and some for testing
cvp = cvpartition(numWords,'HoldOut',0.01);
dataTrain = data(training(cvp),:);
dataTest = data(test(cvp),:);
%change words in the training set of data to word vectors
wordsTrain = dataTrain . Word;
XTrain = word2vec(emb,wordsTrain);
YTrain = dataTrain . Label;

%train a support vector machine classifier to classify the word vectors into positive and negative
model = fitcsvm(XTrain,YTrain);

%change words in the test set of data to word vectors
wordsTest = dataTest.Word;
XTest = word2vec(emb,wordsTest);
YTest = dataTest.Label;
%sentiment labels of test word vectors
[YPred,scores] = predict(model,XTest);

filename = "test.csv";
tbl = readtable (filename, 'TextType', 'string');
textData = tbl.tweet;
textData(1:10);

%use visualisation to demonstrate the accuracy matrix
figure
confusionchart(YTest,YPred);

%apply word embeddings sentiment classifer - apply to test sentences
idx = ~ismember(emb,sents . Vocabulary);
removeWords(sents, idx);
sentimentScore = zeros(size(sents));
for ii = 1 : sents . length
    docwords = sents (ii) . Vocabulary;
    vec = word2vec(emb,docwords);
    [~,scores] = predict(model,vec);
    sentimentScore(ii) = mean(scores(:,1));
    if isnan(sentimentScore(ii))
        sentimentScore(ii) = 0;
    end
    fprintf('Sent: % d, words: % s, FoundScore: % d, GoldScore: % d \n', ii, joinWords(sents(ii)), sentimentScore(ii), actualScore(ii));
end

function data = readLexicon
%read the positive words
fidPositive = fopen(fullfile('opinion-lexicon-English','positive-words.txt'));
C = textscan(fidPositive,'%s','CommentStyle',';');
wordsPositive = string(C{1});
%read the negative words
fidNegative = fopen(fullfile('opinion-lexicon-English','negative-words.txt'));
C = textscan(fidNegative,'%s','CommentStyle',';');
wordsNegative = string(C{1});
fclose all;
%make a table of the labeled words
words = [wordsPositive;wordsNegative];
labels = categorical(nan(numel(words),1));
labels(1:numel(wordsPositive)) = "Positive";
labels(numel(wordsPositive)+1:end) = "Negative";

data = table(words,labels,'VariableNames',{'Word','Label'});

end