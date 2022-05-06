{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE LambdaCase #-}
 
import Object
import Parser
import Runtime
import Audio
import Interface

import Text.Parsec

import System.Console.Haskeline

import Control.Concurrent
import Control.Monad.Trans
import Control.Monad.State
import Control.Lens

import Foreign.Ptr

import Sound.PortAudio

import qualified Data.Set as Set


-- T R O P I C A L
-- A live coding environment for audio synthesis

main = do
  readyFlag <- newEmptyMVar 
  rt <- initRuntime $ fromIntegral framesPerBuffer
  bufferAddress <- getBufferAddress rt
  forkIO $ compute rt readyFlag
  let callback = Just $ mainCallback bufferAddress readyFlag
      fincallback = Just $ putStrLn "stream closed"
      inputLoop = runInputT defaultSettings loop
      hRuntime = Runtime 
        { _env = Set.empty
        , _runtime = rt
        }
  initialize -- initialize portaudio runtime 
  res <- openDefaultStream 0 2 sampRate (Just framesPerBuffer) callback fincallback
  case res of
    Left err -> print err
    Right strm -> do
      startStream strm
      runStateT inputLoop hRuntime
      stopStream strm
      closeStream strm
      return ()
  terminate
  deleteRuntime rt
  return ()

{--
processLine
  :: String
  -> InputT (StateT Runtime IO) ()
processLine input = 
  let x = parse parseAny "" input 
  in case x of
    Right a -> do
      -- here is where type checking should happen
      currentRT <- lift get
      crepr <- liftIO $ generateAST (currentRT^.env) a
      let v = Var 
                { _name = Name ""
                , _this = castPtr crepr
                , _obj = Expression a
                }
      lift $ put $ over env (Set.insert v) currentRT 
      evaluated <- liftIO $ evalc crepr
      outputStrLn $ show evaluated
    Left b -> outputStrLn $ show b
--} 
handleInput
  :: String
  -> InputT (StateT Runtime IO) ()
handleInput input = do 
  let p = parse undefined "" input
  either handleError handleParse p
  where handleError = outputStrLn . show


loop :: InputT (StateT Runtime IO) ()
loop = do
  minput <- getInputLine "🌴 "
  case minput of
    Nothing -> return ()
    Just "quit" -> return ()
    Just input -> do handleInput input
                     loop


-- END