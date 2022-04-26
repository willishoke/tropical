{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE LambdaCase #-}
 
import Audio
import Expression
import Parser
import Runtime

import Text.Parsec
import System.Console.Haskeline
import Control.Concurrent
import Control.Monad
import Control.Monad.Trans
import Control.Monad.State


import Sound.PortAudio.Base
import Sound.PortAudio

-- T R O P I C A L
-- A live coding environment for audio synthesis

-- audio processing code heavily inspired by examples from
-- Portaudio Haskell bindings at
-- https://github.com/sw17ch/portaudio


processLine
  :: String
  -> InputT (StateT (MVar Phases) IO) ()
processLine input = let x = parse line "" input in 
  case x of
    Right a -> do
      crepr <- liftIO $ generateAST a
      evaluated <- liftIO $ evalc crepr
      outputStrLn $ show evaluated
    Left b -> outputStrLn $ show b

loop :: InputT (StateT (MVar Phases) IO) ()
loop = do
  minput <- getInputLine "🍌 "
  case minput of
    Nothing -> return ()
    Just "quit" -> return ()
    Just input -> do processLine input
                     loop


main = do
  initState <- newMVar (Phases 0 0)
  let callback = Just $ mainCallback initState
      fincallback = Just $ putStrLn "stream closed"

  let inputLoop = runInputT defaultSettings loop

  initialize
  res <- openDefaultStream 0 2 sampRate (Just framesPerBuffer) callback fincallback
  case res of
    Left err -> print err
    Right strm -> do
      startStream strm
      runStateT inputLoop initState
      stopStream strm
      closeStream strm
      terminate
      return ()

-- END