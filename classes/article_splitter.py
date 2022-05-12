class ArticleSplitter():
    sentence_max: int = 128

    def __init__(self, sentence_max: int = 128):
        super(ArticleSplitter, self).__init__()

        self.sentence_max = min(self.sentence_max, sentence_max) - 2

    def split(self, articles: list) -> list:
        article_sentences = []
        for article in articles:
            article_split = article.strip().split()
            sentences = []
            for i in range(0, len(article_split), self.sentence_max):
                sentences.append(
                    ' '.join(
                        article_split[i:min(i + self.sentence_max, len(article_split))])
                )
            article_sentences.append(sentences)

        return article_sentences
