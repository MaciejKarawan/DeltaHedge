from pathlib import Path
import json

# 📁 Katalog, w którym zapiszesz pliki (np. bieżący katalog projektu)
output_dir = Path("prezentacja_rise")
output_dir.mkdir(exist_ok=True)

# 📘 Zawartość notebooka jako Python dict
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {"slideshow": {"slide_type": "slide"}},
            "source": [
                "# 📊 Moja prezentacja\n",
                "## Interaktywne wykresy i LaTeX w Jupyter + RISE\n",
                "**Autor:** Misiaczek 💛"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"slideshow": {"slide_type": "subslide"}},
            "source": [
                "### Wzór kwadratowy\n",
                "\n",
                "$$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"slideshow": {"slide_type": "subslide"}},
            "outputs": [],
            "source": [
                "# 📈 Wykres z Plotly\n",
                "import plotly.express as px\n",
                "\n",
                "df = px.data.iris()\n",
                "fig = px.scatter(df, x=\"sepal_width\", y=\"sepal_length\", color=\"species\",\n",
                "                 title=\"Iris Dataset: Sepal Width vs Length\")\n",
                "fig.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {"slideshow": {"slide_type": "subslide"}},
            "source": [
                "### ✅ Podsumowanie\n",
                "\n",
                "- LaTeX działa dzięki MathJax\n",
                "- Plotly tworzy interaktywne wykresy\n",
                "- RISE zamienia notebooka w prezentację\n",
                "- Prezentację można zapisać jako HTML"
            ]
        }
    ],
    "metadata": {
        "celltoolbar": "Slideshow",
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {"name": "python"},
        "rise": {
            "transition": "fade",
            "start_slideshow_at": "beginning",
            "slideNumber": True,
            "controls": True,
            "progress": True
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

# 📜 Plik requirements.txt
requirements_txt = """notebook
rise
plotly
"""

# 📝 Zapisz pliki
notebook_path = output_dir / "prezentacja.ipynb"
requirements_path = output_dir / "requirements.txt"

notebook_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
requirements_path.write_text(requirements_txt, encoding="utf-8")

print("✅ Gotowe! Znajdziesz pliki w folderze:", output_dir.resolve())