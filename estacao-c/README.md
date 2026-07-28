# 🧠 Módulo de Neuroimagem Computacional

▶️ Antes de tudo, assista à [apresentação](https://canva.link/nufhpt5l1hjtmk6)!

Nesse repositório você encontrará quatro pastas que contém:

- 📁 **`brains/`**:
    - Superfícies corticais de diversas espécies em formato `*.stl`
    - Uma subpasta referente a um dado de má qualidade. Dentro da pasta, além da superfície, existe imagens do tipo `*.nii.gz` 
- 📁 **`melted-sheets/`**:
    - Uma série de subdiretórios cada um com um arquivo `*.csv` referente a volres numéricos do processo de derretimento (**melting**) dos exemplos de superfícies corticais
- 📁 **`melting-videos/`**:
    - Vídeos demonstrativos da aplicação do _pipeline_ de derretimento cortical em 3 espécies:
        - 🐬 Boto-cinza (_Sotalia guianensis_)
        - 🦭 Leão marinho (_Eumetopias jubatus_)
        - 💆 Humano (_Homo sapiens_)
- 📁 **`notebooks/`**:
    - Jupyter Notebooks (`.ipynb`) com rotinas práticas de processamento e análise dos dados corticais.

Abra os **notebooks** no seu editor favorito e siga as intruções das atividades! 


---

# 🛠️ Pré-requisitos

## 💻 Editores

Para executar e visualizar os *notebooks*, sugerimos o uso de um dos ambientes abaixo:

1. **[VS Code](https://code.visualstudio.com/download?_exp_download=fb315fc982)** *(Recomendado)* — Instale também a extensão oficial **[Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)**.
2. **[Jupyter Lab / Notebook](https://jupyter.org/install)**
3. **[Google Colab](https://colab.research.google.com/)** *(Alternativa sem necessidade de instalação local)*


## 📦 Softwares e Bibliotecas

Certifique-se de ter os seguintes programas gratuitos instalados:

1. **[Python](https://www.python.org/downloads/)**
2. **[MeshLab](https://www.meshlab.net/#download)** *(para visualização e manipulação das malhas 3D `.stl`)*

### ⚡ Instalação das dependências Python

Após instalar o Python, abra o terminal e execute o comando abaixo para instalar as bibliotecas necessárias:

```bash
pip install numpy matplotlib scipy pandas nibabel trimesh pyvista pymeshlab robust-laplacian
```