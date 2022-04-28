{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE LambdaCase #-}
 
import Expression
import Object
import Parser
import Runtime

import Text.Parsec
import System.Console.Haskeline
import Control.Concurrent
import Control.Monad
import Control.Monad.Trans
import Control.Monad.State

import Foreign.Ptr

import qualified Data.Set as Set

import Sound.PortAudio.Base
import Sound.PortAudio

-- T R O P I C A L
-- A live coding environment for audio synthesis

main = do
  readyFlag <- newEmptyMVar 

  let callback = Just $ mainCallback nullPtr readyFlag
      fincallback = Just $ putStrLn "stream closed"
      inputLoop = runInputT defaultSettings loop

  initialize -- initialize portaudio runtime 
  res <- openDefaultStream 0 2 sampRate (Just framesPerBuffer) callback fincallback
  case res of
    Left err -> print err
    Right strm -> do
      startStream strm
      runStateT inputLoop Set.empty
      stopStream strm
      closeStream strm
      terminate
      return ()

processLine
  :: String
  -> InputT (StateT Env IO) ()
processLine input = 
  let x = parse line "" input 
  in case x of
    Right a -> do
      -- here is where type checking should happen
      env <- lift get
      crepr <- liftIO $ generateAST env a
      --put $ Set.insert a env
      evaluated <- liftIO $ evalc crepr
      outputStrLn $ show evaluated
    Left b -> outputStrLn $ show b

loop :: InputT (StateT Env IO) ()
loop = do
  minput <- getInputLine "🌴 "
  case minput of
    Nothing -> return ()
    Just "quit" -> return ()
    Just input -> do processLine input
                     loop


-- END