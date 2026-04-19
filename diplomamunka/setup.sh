#!/bin/bash
# ELTE FI elteikthesis.cls és logó letöltése
# Forrás: https://github.com/mcserep/elteikthesis

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Sablon letöltése..."
curl -sL "https://raw.githubusercontent.com/mcserep/elteikthesis/master/elteikthesis.cls" \
     -o "$DIR/elteikthesis.cls"

echo "ELTE logó letöltése..."
curl -sL "https://raw.githubusercontent.com/mcserep/elteikthesis/master/images/elte_cimer_szines.pdf" \
     -o "$DIR/images/elte_cimer_szines.pdf" 2>/dev/null || \
curl -sL "https://raw.githubusercontent.com/mcserep/elteikthesis/master/images/elte_cimer_szines.png" \
     -o "$DIR/images/elte_cimer_szines.png" 2>/dev/null || \
echo "  FIGYELEM: Logót kézzel kell beszerezni: images/elte_cimer_szines.{pdf,png}"

echo ""
echo "Fordítás:"
echo "  cd $DIR"
echo "  pdflatex main.tex"
echo "  biber main"
echo "  pdflatex main.tex"
echo "  pdflatex main.tex"
echo ""
echo "Vagy Overleaf-en: https://www.overleaf.com/latex/templates/elte-fi-thesis-template/scjzzzbjvwfz"
