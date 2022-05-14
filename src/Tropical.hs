{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE LambdaCase #-}
 
import Parser
import Runtime
import Audio
import Interface

import Text.Parsec

import System.Console.Haskeline

import Control.Concurrent
import Control.Monad.Extra (loopM)
import Control.Monad.Trans
import Control.Monad.State
import Control.Lens

import Foreign.Ptr

import Sound.PortAudio

import qualified Data.Map as Map


-- T R O P I C A L
-- A live coding environment for audio synthesis

main :: IO ()
main = do
  readyFlag <- newEmptyMVar 
  rtPtr <- initRuntime $ fromIntegral framesPerBuffer
  bufferAddress <- getBufferAddress rtPtr
  forkIO $ compute rtPtr readyFlag
  let callback = Just $ mainCallback bufferAddress readyFlag
      fincallback = Just $ putStrLn "stream closed"
      rt = Runtime 
        { _env = Env Map.empty
        , _runtime = rtPtr
        }
  initialize -- initialize portaudio runtime 
  res <- openDefaultStream 0 2 sampRate (Just framesPerBuffer) callback fincallback
  case res of
    Left err -> print err
    Right strm -> do
      startStream strm
      flip runStateT rt $ runInputT defaultSettings loop
      stopStream strm
      closeStream strm
      pure ()
  terminate
  deleteRuntime rtPtr
  pure ()

handleInput
  :: String
  -> InputT (StateT Runtime IO) ()
handleInput input = do 
  let p = parse undefined "" input
  either handleError handleParse p
  where handleError = outputStrLn . show


loop :: InputT (StateT Runtime IO) ()
loop = getInputLine "🌴 " >>= \x -> 
  case x of
    Nothing -> pure ()
    Just "quit" -> pure ()
    Just input -> do
      handleInput input
      loop


-- END