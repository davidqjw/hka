from setuptools import setup, find_packages

with open("requirements.txt",encoding='utf-8') as fp:
    requirements = fp.read().splitlines()

extras_require = {
    'core': requirements,
    'retriever': ['pyserini', 'sentence-transformers>=3.0.1'],
    'generator': ['vllm'],
    'multimodal': ['timm', 'torchvision', 'pillow', 'qwen_vl_utils']
}
extras_require['full'] = sum(extras_require.values(), [])

setup(
    name="hokieknowledgeagent",
    packages=find_packages(),
    author="",
    description="Virginia Tech Hokie Knowledge Agent",
    long_description_content_type="text/markdown",
    package_data={"": ["*.yaml"]},
    include_package_data=True,
    install_requires=extras_require['core'],
    extras_require=extras_require,
    python_requires=">=3.9",
)
