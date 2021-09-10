# dog-breed-recognition-v3
### Mini-projeto de Machine Learning/Computer Vision: 
### "Sistema de reconhecimento de raças de cachorro"

![image](https://user-images.githubusercontent.com/19409288/132823095-503ca657-5032-448c-8212-6b7f14b7893f.png)

## A ideia
Basicamente a ideia é termos um sistema simples de reconhecimento de raças de
cachorro. Porém, há uma dificuldade extra, pois estamos interessados em adicionar novas
raças que ainda não foram vistas no tempo de treinamento, e saber identificar unknowns
(como, por exemplo, nossos queridos vira-latas).

## Outline da solução

Como dataset, o candidato deve utilizar o seguinte:

https://drive.google.com/file/d/1DAyRYzZ9B-Nz5hLL9XIm3S3kDI5FBJH0/view?usp=sharing

## Primeira Parte

Na pasta **train** está um total de 100 raças de cachorros, uma em cada pasta. Num primeiro
momento deve ser desenvolvido um sistema que, dada uma foto de um cachorro destas
raças, consiga dizer qual é a sua raça e com uma confiança desta predição. Como
referência, obtivemos uma acurácia média por classe em torno de 80% com modelos
baseados em backbones populares na literatura (sem fine-tunning).

## Segunda Parte

Agora deve-se criar um sistema para lidar com raças não vistas em tempo de treinamento,
ou seja, raças diferentes das 100 utilizadas inicialmente. Como no sistema anterior, ele
deve ser treinado utilizando-se as imagens da pasta **train**.

Para permitir o reconhecimento de novas raças, dentro da pasta **recognition**, temos uma
pasta **enroll** com fotos de outras raças. Estas fotos deverão ser utilizadas para ensinar ao
sistema como determinada raça se parece. Na hora do enroll, os labels devem ser
utilizados. Por exemplo, adicionaremos 5 fotos da raça Bulldog no sistema, informando a
sua raça, sendo que Bulldog não estava presente no treino. A partir de agora o sistema
deve conseguir reconhecer esta raça.

Na pasta **recognition/test** temos algumas fotos destas 20 raças para o sistema
reconhecer, onde os labels vão servir só para conferir a acurácia do sistema.
O processo de enroll deve ser rápido (menos que 1 segundo por imagem), pois não
queremos um downtime grande para o cliente. Também não é necessário reconhecer as
100 raças do treinamento, apenas as do processo de enroll.

Pode-se pensar que este sistema esteja rodando em algum cliente, não sendo possível
definir de antemão quais raças de cachorros podem aparecer. Porém, ele pode fazer o
cadastro (enroll) de novas raças e, com isso, reconhecê-las.

## Terceira Parte

Além disso, o candidato deve pensar sobre o que acontece se uma raça previamente não
vista, e nem cadastrada, é fornecida ao sistema. Como o sistema pode agir para identificar
que essa raça não existe no banco de dados? Ou seja, se o cliente passar uma foto de um
Golden Retriever cuja raça não foi cadastrada no processo de enroll, o sistema deve
informar que essa não é uma raça cadastrada. Esse problema é o problema de
classificação de unknowns. _Não são fornecidas imagens para esse teste, mas se espera do
candidato que ele consiga de alguma forma avaliar quantitativamente os métodos empregados
aqui._

## Entregáveis

O candidato pode utilizar as ferramentas que achar mais convenientes, como Keras,
Pytorch, etc. Deve-se ter uma maneira simples de rodar novamente a etapa de
treinamento e testes. Pode ser organizado tanto em scripts shell, Python, R, etc. Docker
também pode ser utilizado para facilitar a criação de um ambiente de testes, porém não é
necessário caso a instalação dos requirements seja simples. Por fim, recomenda-se
organizar o código no github para facilitar a avaliação.

É desejável que se tenha algumas entregas para discussão de como foi solucionado o
problema:
- Qual acurácia do sistema e quais métricas foram utilizadas para verificar a sua qualidade?
- Como foi definido _unknowns_ no sistema, e como foi feito teste para identificá-los? Quais
métricas foram utilizadas?

## Solução

Um resumo das soluções pode ser conferido em: **Solução.pdf**

https://github.com/amandaleonel/dog-breed-recognition-v3/blob/main/Solu%C3%A7%C3%A3o.pdf
