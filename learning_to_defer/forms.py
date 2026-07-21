from django import forms


class HumanExpertLabelForm(forms.Form):
    label = forms.ChoiceField(
        label="Choose the article category",
        choices=[
            ("0", "World"),
            ("1", "Sports"),
            ("2", "Business"),
            ("3", "Sci/Tech"),
        ],
        widget=forms.RadioSelect,
    )
class ArticleClassificationForm(forms.Form):
    text = forms.CharField(
        label="Paste a news article",
        max_length=10000,
        widget=forms.Textarea(
            attrs={
                "rows": 8,
                "placeholder": "Paste the article text here...",
            }
        ),
    )