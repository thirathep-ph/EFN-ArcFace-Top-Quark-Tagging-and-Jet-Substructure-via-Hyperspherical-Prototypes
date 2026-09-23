from setuptools import setup, find_packages

setup(
    name="arcefn",
    version="0.1.0",
    python_requires=">=3.10, <3.13",
    packages=find_packages(),
    package_data={'arcefn': ['config.json']},
    install_requires=[
        "torch",
        "numpy",
        "awkward",
        "vector",
        "h5py",
        "hdf5plugin",
        "fastjet",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "umap-learn",
    ],
    entry_points={
        'console_scripts': [
            'arcefn-train=scripts.train_arcface:main',
            'arcefn-evaluate=scripts.evaluate_model:main',
            'arcefn-baselines=scripts.train_baselines:main',
        ],
    },
    author="Thirathep Naowabut Phiankham",
    description="EFN-ArcFace Top Quark Tagging and Jet Substructure via Hyperspherical Prototypes",
)
