from setuptools import setup

setup(
    name='shitsu',
    version='0.1.0',    
    description='A fast multilingual text quality classifier',
    url='https://github.com/lightblue-tech/shitsu',
    author='Peter Devine',
    author_email='peter@lightblue-tech.com',
    license='MIT',
    packages=['shitsu'],
    install_requires=['fasttext',
                      'torch',
                      'safetensors',
                      'tqdm',
                      'huggingface_hub'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Programming Language :: Python :: 3',
    ],
)
