class QueryResult:
    def __init__(self, meta_context: str, context: str):
        self.meta_context = meta_context
        self.context = context

    def get(self):
        return self.meta_context, self.context

    def get_meta_context(self):
        return self.meta_context

    def get_context(self):
        return self.context
