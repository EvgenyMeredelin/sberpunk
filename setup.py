from setuptools import find_namespace_packages, setup


setup(
    name='sberpunk',
    version='0.2',
    packages=find_namespace_packages(where='src'),
    package_dir={'': 'src'},
    package_data={'sberpunk.examples': ['*', '*/*']},
    python_requires='>=3.10, <4',
    install_requires=[
        'black', 'gensim', 'matplotlib', 'more-itertools',
        'multimethod', 'nltk', 'numpy', 'pandas', 'pymorphy3',
        'scipy==1.12', 'seaborn', 'sentence_transformers',
        'torch==2.3.1', 'transformers'
    ],
    extras_require={
        'directory_scanner': [
            'IPython', 'docx2python', 'extract_msg', 'openpyxl',
            'patool', 'pdfminer.six', 'pillow', 'pytesseract',
            'python-pptx', 'striprtf', 'tabulate', 'xlrd'
        ]
    },
    author='Evgeny Meredelin',
    author_email='eimeredelin@sberbank.ru'
)