;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "TT"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "12pt" "a4paper")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T2A") ("babel" "russian") ("lmodern" "") ("mathtext" "") ("microtype" "") ("amsmath" "") ("amssymb" "") ("graphicx" "") ("booktabs" "") ("xcolor" "") ("hyperref" "") ("tikz" "") ("float" "") ("enumitem" "") ("listings" "") ("placeins" "") ("geometry" "left=2cm" "right=2cm" "top=2cm") ("titlesec" "")))
   (add-to-list 'LaTeX-verbatim-environments-local "lstlisting")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "lstinline")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art12"
    "inputenc"
    "fontenc"
    "babel"
    "lmodern"
    "mathtext"
    "microtype"
    "amsmath"
    "amssymb"
    "graphicx"
    "booktabs"
    "xcolor"
    "hyperref"
    "tikz"
    "float"
    "enumitem"
    "listings"
    "placeins"
    "geometry"
    "titlesec"))
 :latex)

