import Data.List

type Network = [[[Double]]]

makeList :: Int -> [Double]
makeList 0 = [1.0]
makeList x = 1.0 : (makeList (x-1))

makeMatrix :: Int -> Int -> [[Double]]
makeMatrix 0 y = []
makeMatrix x y = ((makeList y) : (makeMatrix (x-1) y))

--Recebe uma lista com o numero de neuronios por layer e devolve uma lista de matrizes
--com os pesos das ligacoes
createNetwork :: [Int] -> Network
createNetwork [x]       = []
createNetwork (x:y:ys) = (makeMatrix y x) : (createNetwork (y:ys))

--funcao de ativacao sigmoide
sigmoid :: Double -> Double -> Double
sigmoid x a = 1/(1 + exp(-a*x))

ativacao :: Double -> Double
ativacao a | a < 2 = 0
           | otherwise = 1

-- Dados vetor de entradas, matriz de pesos e função de ativação
-- retorna a lista de output da camada
calculaAct :: [Double] -> [[Double]] -> (Double->Double) -> [Double]
calculaAct xs [] f      = []
calculaAct xs (y:ys) f  = [f $ sum $ zipWith (*) xs y] ++ calculaAct xs ys f

-- Dados vetor de entradas, rede neural e função de ativação
-- retorna a lista de output e pesos da camada
calcOutput :: [Double] -> Network -> (Double -> Double) -> [[Double]]
calcOutput xs [] f     = []
calcOutput xs (r:rs) f = [resultCamada] ++ calcOutput resultCamada rs f
    where
        resultCamada = calculaAct (xs++[-1.0]) r f

--delta para um neuron da camada de saida
outputNeuronDeltas :: Double -> Double -> Double
outputNeuronDeltas target output = -(target-output) * output * (1-output)

--delta de todos os neurons da camada de saida
outputLayerDeltas :: [Double] -> [Double] -> [Double]
outputLayerDeltas targets output = [outputNeuronDeltas t o | (t, o) <- zip targets output]

--delta de um neuron de uma camada escondida
neuronDelta :: [Double] -> [Double] -> Double -> Double
neuronDelta weights prevDeltas output = (sum [w*g | (w, g) <- zip weights prevDeltas]) * output * (1-output)

--delta de todos os neurons de uma camada escondida
layerDeltas :: [[Double]] -> [Double] -> [Double] -> [Double]
layerDeltas weights prevDeltas outputs = [neuronDelta w prevDeltas s | (w, s) <- zip neurons outputs]
                                          where neurons = [[neuron !! n | neuron <- weights] | n <- [0..length (head weights)-1]]

--calcula recursivamente o delta para todas as camadas
hiddenLayersDeltas :: Network -> [[Double]] -> [Double] -> [[Double]]
hiddenLayersDeltas _ [] _ = []
hiddenLayersDeltas net outputs lastDeltas = (hiddenLayersDeltas (tail net) (tail outputs) layerDelta) ++ [layerDelta]
                                            where layerDelta = layerDeltas (head net) lastDeltas (head outputs)

--delta da rede inteira, inverte a lista e a rede para ir de tras para frente
networkDeltas :: Network -> [[Double]] -> [Double] -> [[Double]]
networkDeltas net outputs targets = (hiddenLayersDeltas revNet (tail revOutput) outputDeltas) ++ [outputDeltas]
                                  where revOutput = reverse outputs
                                        revNet = reverse net
                                        outputDeltas = outputLayerDeltas targets (head revOutput)

--gradientes de um neuron -> delta * output
neuronGradients :: [Double] -> Double -> [Double]
neuronGradients outputs delta = (map (delta*) outputs) ++ [(-delta)]

--gradientes de todos os neurons da camada
layerGradients :: [Double] -> [Double] -> [[Double]]
layerGradients outputs deltas = map (neuronGradients outputs) deltas

--gradientes de toda a rede
calcGradients :: [[Double]] -> [[Double]] -> [[[Double]]]
calcGradients [] _ = []
calcGradients (d:ds) (o:os) = (layerGradients o d) : (calcGradients ds os)


--Recebe a rede, o gamma os gradientes ajusta os pesos:
--Calcula a variacao
--Ajusta os pesos
ajustaPesos :: Network -> Double -> Network -> Network
ajustaPesos net gamma grad = [[zipWith (+) x y | (x, y) <- zip xs ys] | (xs, ys) <- zip net term]
                           where
                             term = map (map (map (* (-gamma)))) grad

--Faz o treinamento para uma passagem do conjunto de dados
passOnce :: Network -> [[Double]] -> [[Double]] -> Double -> Network
passOnce net [] [] _ = net
passOnce net (i:is) (t:ts) g = passOnce (ajustaPesos net g gradients) is ts g
                    where
                      outputs   = calcOutput i net (sigmoid 1)
                      deltas    = networkDeltas net outputs t
                      gradients = calcGradients deltas (i:outputs)

--Faz n passagens pelo conjunto de dados
training :: Network -> [[Double]] -> [[Double]] -> Double -> Double -> Network
training net _ _ _ 0 = net
training net inputs targets gamma step = training (passOnce net inputs targets gamma) inputs targets gamma (step-1)

--cria rede
net = createNetwork [2,5,1]

--inputs -> numeros de [0.0,0.2] [0.0,0.4]
--targets -> soma dos numeros
x = [[a, b]| a <- [0..20], b <- [20..40]] :: [[Double]]
y = [[sum a] | a <- x] :: [[Double]]

inputs = map (map (/100)) x
targets = map (map (/100)) y

gamma = 0.1
step = 200

redeTreinada = training net inputs targets gamma step

resultado = calcOutput [0.5, 0.3] treinamento (sigmoid 1)