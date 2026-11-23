from rest_framework import serializers


class AskSerializer(serializers.Serializer):
    question = serializers.CharField(max_length=4000)
    mode = serializers.ChoiceField(choices=["hybrid", "dense", "bm25"], default="hybrid")
    rerank = serializers.BooleanField(default=True)
    stream = serializers.BooleanField(default=False)


class TextSerializer(serializers.Serializer):
    text = serializers.CharField(max_length=4000)


class GenerateSerializer(serializers.Serializer):
    prompt = serializers.CharField(max_length=2000)
    max_new = serializers.IntegerField(min_value=1, max_value=120, default=60)
    temperature = serializers.FloatField(min_value=0.05, max_value=2.0, default=0.8)
    top_k = serializers.IntegerField(min_value=0, max_value=50, default=0)
    top_p = serializers.FloatField(min_value=0.05, max_value=1.0, default=1.0)
    greedy = serializers.BooleanField(default=False)
