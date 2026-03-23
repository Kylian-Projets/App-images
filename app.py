import streamlit as st
import numpy as np
import processing
from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps
import matplotlib.pyplot as plt


default_image_path = "data/default_image.jpg"

st.set_page_config(layout="wide",page_title="Image Processing")
st.title("Traitement d'images")


uploaded_file = st.file_uploader("Choisissez une image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
else:
    img = Image.open(default_image_path)
    st.info("Utilisation de l'image par défaut")



st.container(height=20,border=False)

st.write(f"Format de l'image : {img.mode}")
width,height = img.size
st.write(f"Largeur : {width} pixels")
st.write(f"Hauteur : {height} pixels")
st.container(height=20,border=False)

if img is not None:

    # color channels

    st.subheader("Extraction des 3 canaux de couleur (Rouge,Vert,Bleu)")

    img.thumbnail((500,500))
    r,g,b = img.split()
    s,red,green,blue = st.columns([1,1,1,1])
    with s:
        st.subheader("Image source")
        st.image(img)
    with red:
        st.subheader("Canal rouge")
        st.image(r)
    with green:
        st.subheader("Canal vert")
        st.image(g)
    with blue:
        st.subheader("Canal bleu")
        st.image(b)
    
    # Gray scale 
    st.container(height=20,border=False)
    st.subheader("Passage de l'image source en niveau de gris")

    img_gray = img.convert("L")
    gray_scale = st.slider("Ajustez le niveau de gris avec les commandes ci-dessous :", min_value=2, max_value=255, value=128)
    st.container(height=20,border=False)

    col_empty1,col_gray_image,col_hist,col_empty2 = st.columns([1,1,1,1])

    img_quantified = img_gray.quantize(colors=gray_scale)
    with col_gray_image:
        st.image(img_quantified, caption=f"Image source en {gray_scale} niveaux de gris",use_container_width=True)
    with col_hist:
        st.pyplot(processing.histogram(img_quantified,gray_scale))
    
    # Blur
    st.container(height=20,border=False)
    st.subheader("Floutter une image")
    st.write("Utilisation de 3 types de flou différents")

    col_source,col_slider_blur1,col_slider_blur2,col_slider_blur3 = st.columns([1,1,1,1])

    with col_source:
        st.container(height=69,border=False)
        st.image(img,caption=("Image source"))
    with col_slider_blur1:
        gaussian_blur_slider = st.slider("Choisissez l'intensité du filtre Gaussien",min_value=0,max_value=20,value=2)
        st.image(img.filter(ImageFilter.GaussianBlur(gaussian_blur_slider)),caption=f"Filtre Gaussien avec une intensité de {gaussian_blur_slider}")
        st.write("Le flou Gaussien réduit les détails de l'image en utilisant une fonction mathématique 'en cloche' (la courbe de Gauss). Contrairement à un flou classique," \
        " il accorde plus d'importance aux pixels centraux qu'aux pixels éloignés, ce qui donne un rendu très naturel et fluide.")
    with col_slider_blur2:
        mean_blur_slider = st.slider("Choisissez l'intensité du filtre moyen",min_value=0,max_value=25,value=2)
        st.image(img.filter(ImageFilter.BoxBlur(mean_blur_slider)),caption=f"Filtre moyen avec une intensité de {mean_blur_slider}")
        st.write("Ce filtre remplace la couleur de chaque pixel par la moyenne arithmétique de tous les pixels voisins. Plus le rayon est grand, plus l'image semble 'diluée'." \
        " C'est un filtre simple et rapide, mais il peut donner un aspect un peu plus rigide et géométrique que le flou Gaussien.")
    with col_slider_blur3:
        median_blur_slider = st.slider("Choisissez l'intensité du filtre médian",min_value=1,max_value=31,value=1,step=2)
        st.image(img.filter(ImageFilter.MedianFilter(median_blur_slider)),caption=f"Filtre médian avec une intensité de {median_blur_slider}")
        st.write("Le flou médian ne fait pas de calcul de moyenne : il trie les pixels voisins et choisit la valeur centrale (la médiane). " \
        "Son grand pouvoir est de supprimer le bruit numérique (comme les petits points blancs ou noirs) tout en préservant la netteté des bords." \
        " C'est le filtre préféré pour nettoyer des images dégradées sans les rendre totalement floues.")
    
    # histogram equalization
    st.container(height=20,border=False)
    st.subheader("Égalisation d'histogramme (en niveaux de gris)")
    st.write("Ce traitement étire les contrastes. Si un image est trop sombre ou trop grise," \
    "il redistribue les intensités sur toute la plage (0-255).")
    st.write("EXEMPLE :")
    
    img_gray_hist = img.convert("L").quantize(colors=255)
    img_hist_eq = ImageOps.equalize(img_gray_hist.convert("L"))
    _,col_gray_hist, col_hist_eq,_ = st.columns([1,1,1,1])

    with col_gray_hist:
        st.image(img_gray_hist,caption="Image en 255 niveaux de gris",use_container_width=True)
    with col_hist_eq:
        st.image(img_hist_eq,caption="Image sur 255 niveaux de gris avec histogramme égalisé",use_container_width=True)
    
    # binarization
    st.container(height=20,border=False)
    st.subheader("Binarisation (passage en noir et blanc)")
    st.write("Binarisation de l'image en niveaux de gris utilisée pour l'exemple précédent ")

    _,col_bin_manual,col_bin_otsu,_ = st.columns([1,1,1,1])

    with col_bin_manual:
        threshold = st.slider("Choisissez un seuil de binarisation", min_value=0,max_value=255,value=128)
        Lut = [0 if i < threshold else 255 for i in range(256)]
        img_bin = img_gray_hist.convert("L").point(Lut, 'L')
        st.image(img_bin,caption=f"Image binarisée avec un seuil de {threshold}",use_container_width=True)
        st.write("C'est la méthode la plus courante. L'avantage est que c'est très rapide mais ce traitement dépend énormément de l'éclairage.")
    with col_bin_otsu:
        opt_threshold,img_bin_otsu = processing.otsu(img_gray_hist)
        st.slider("Valeur de seuil optimal calculée par l'algorithme d'Otsu",value=opt_threshold,min_value=0.0,max_value=255.0,disabled=True)
        st.image(img_bin_otsu,caption="Image binarisée avec l'algorithme d'Otsu",use_container_width=True)
        st.write("La méthode d'Otsu est dite 'intelligente'. Elle analyse l'histogramme de l'image pour trouver mathématiquement le seuil qui sépare" \
        " le mieux le premier plan de l'arrière-plan.")

    # Morphological Transformations
    st.container(height=20,border=False)
    st.subheader("Opérations morphologiques")
    st.write("Ces procédés sont basés sur la forme de l'image et s'appliquent généralement sur des images binaires (noir et blanc). Ils utilisent deux éléments :" \
    " une image et un noyau (kernel) qui définit la zone d'influence du filtre.")

    kernel_size = st.slider("Choissez la taille du noyau", min_value=1, max_value=15, value=2)
    iterations = st.slider("Choisissez le nombre de fois que le filtre doit être appliqué",min_value=1,max_value=5,value=1)

    col_erode,col_dilate,col_open,col_close = st.columns([1,1,1,1])

    img_erode,img_dilate,img_open,img_close = processing.morpho(img_bin_otsu,kernel_size,iterations)

    with col_erode:
        st.subheader("Érosion")
        st.image(img_erode,caption=f"Érosion avec une noyau de ({kernel_size} x {kernel_size}) efectuée {iterations} fois")
        st.write("L'érosion 'grignote' les bords des objets blancs. Si un pixel blanc a au moins un voisin noir, il devient noir. Utile pour supprimer les petits" \
        " bruits blancs isolés (parasites) ou détacher deux objets qui se touchent à peine.")
    with col_dilate:
        st.subheader("Dilatation")
        st.image(img_dilate,caption=f"Dilatation avec un noyau de ({kernel_size} x {kernel_size}) effectuée {iterations} fois")
        st.write("C'est l'inverse de l'érosion. Elle ajoute des pixels blancs sur les bords des objets. Si un pixel noir a un voisin blanc, il devient blanc." \
        " Utile pour combler des petits trous noirs à l'intérieur d'un objet ou relier des parties brisées d'une même forme.")
    with col_open:
        st.subheader("Ouverture")
        st.image(img_open,caption=f"Ouverture avec un noyau de ({kernel_size} x {kernel_size})")
        st.write("C'est une érosion suivie d'une dilatation. Très efficace pour supprimer le bruit tout en gardant la taille originale de l'objet")
    with col_close:
        st.subheader("Fermeture")
        st.image(img_close, caption=f"Fermeture avec un noyau de ({kernel_size} x {kernel_size})")
        st.write("C'est une dilatation suivie d'une érosion. Parfait pour boucher les petits points noirs ou les fissures au sein d'un objet blanc.")


    # Contour detection
    st.container(height=20,border=False)
    st.subheader("Détection de contours")
    st.write("Pour détceter les contours sur une image elle doit être pré-traitée avec certaines méthodes décrites précédemment.")
    st.write("")

    examples_images = {
        "Ticket" : "data/Ticket.jpeg",
        "Perroquet" : "data/default_image.jpg",
        "Étoiles" : "data/etoiles.jpg",
        "Poissons" : "data/poissons.jpg",
        "Voitures" : "data/voitures.jpg",
    }

    choice = st.selectbox("Quelle image souhaitez vous traiter ?", ["Charger mon image"] + list(examples_images.keys()))

    if choice == "Charger mon image":
        file = st.file_uploader("Upload ton image ici", type=["png", "jpg", "jpeg","webp"])
        if file:
            img_source = Image.open(file)
        else:
            img_source = img
    else:
        path = examples_images[choice]
        img_source = Image.open(path)

    col_settings = st.columns(3)
    with col_settings[0]:
        bg = st.radio("Type de fond", ["Clair (ex: papier)", "Foncé (ex: nuit)"], index=0)
        bg_param = "light" if "Clair" in bg else "dark"
    with col_settings[1]:
        blur = st.slider("Force du lissage", 1, 15, 5, step=2)
    with col_settings[2]:
        min_size = st.number_input("Taille min. de l'objet (pixels)", value=100, step=50)

    # processing
    results = processing.full_processing_pipeline(img_source, bg_type=bg_param, blur_strength=blur, min_area=min_size)

    
    tabs = st.tabs(["1. Gris & Flou", "2. Binarisation", "3. Morphologie", "4. Résultat Final"])

    with tabs[0]:
       
        _, col, _ = st.columns([1, 2, 1]) 
        col.image(results["gris"], caption="Image en Gris + Flou", use_container_width=True)

    with tabs[1]:
        _, col, _ = st.columns([1, 2, 1])
       
        col.image(results["bin"], caption="Noir et Blanc brut (Otsu)", use_container_width=True)

    with tabs[2]:
        _, col, _ = st.columns([1, 2, 1])
        col.image(results["morpho"], caption="Image nettoyée (Morphologie)", use_container_width=True)

    with tabs[3]:
        st.metric("Objets identifiés", results["count"])
        _, col, _ = st.columns([1, 3, 1])

        col.image(results["final"], caption="Détection par Bounding Boxes")

        
