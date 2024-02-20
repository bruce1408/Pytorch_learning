from setuptools import setup, find_packages

setup(
    name="printk",
    version="0.1",
    packages=find_packages(),
    description="A colorful print tool for python",
    author="bruce_cui",
    author_email="summer56567@163.com",
    install_requires=[
        # 依赖列表
        # 例如: "requests >= 2.22.0"
        "termcolor >= 2.3.0",
    ],
)
