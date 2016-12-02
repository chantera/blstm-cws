#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import argparse
from collections.abc import Callable
from datetime import datetime
from dateutil.tz import tzlocal
from inspect import getfile
import logging
import os
import signal
import sys
import time
import uuid


class Singleton(metaclass=ABCMeta):
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls.__instance


def _gen_hexid():
    return uuid.uuid4().hex[:6]


class _Formatter(logging.Formatter):

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            t = datefmt.replace('%f', str(int(record.msecs)))
            s = time.strftime(t, ct)
        else:
            t = time.strftime(self.default_time_format, ct)
            s = self.default_msec_format % (t, record.msecs)
        return s


class Logger:
    __instance = None

    FATAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    VERBOSE = 5

    LABELS = {
        ERROR: 'error',
        WARNING: 'warn',
        INFO: 'info',
        DEBUG: 'debug',
        VERBOSE: 'trace',
    }

    # global config
    _verbose = False
    _logdir = './logs'
    _loglevel = INFO
    _format = "%(asctime)-15s\t%(accessid)s\t[%(priority)s]\t%(message)s"
    _dateformat = "%Y-%m-%d %H:%M:%S.%f %Z"

    def __init__(self):
        raise NotImplementedError()

    @classmethod
    def finalize(cls):
        try:
            cls._get_instance()._stop()
        except FileNotFoundError:
            pass
        finally:
            Logger.__instance = None

    def _initialize(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(Logger.VERBOSE)

        logdir = os.path.abspath(os.path.expanduser(Logger._logdir))
        if not os.path.isdir(logdir):
            raise FileNotFoundError("logdir was not found: '%s'" % logdir)
        logfile = logdir + '/' + datetime.now().strftime("%Y%m%d") + '.log'

        file_handler = logging.FileHandler(filename=logfile)
        file_handler.setLevel(Logger._loglevel)
        file_handler.setFormatter(_Formatter(Logger._format, Logger._dateformat))
        self._add_handler(file_handler)

        if Logger._verbose:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(Logger.VERBOSE)
            stream_handler.setFormatter(_Formatter(Logger._format, Logger._dateformat))
            self._add_handler(stream_handler)

        self._start()
        self._initialized = True

    def _add_handler(self, hdlr):
        self._logger.addHandler(hdlr)

    def _remove_handler(self, hdlr):
        hdlr.close()
        self._logger.removeHandler(hdlr)

    @classmethod
    def _get_instance(cls):
        if cls.__instance is None:
            instance = object.__new__(cls)
            instance._initialize()
            cls.__instance = instance
        return cls.__instance

    @classmethod
    def configure(cls, loglevel=_loglevel, verbose=_verbose, logdir=_logdir):
        cls._loglevel = loglevel
        cls._verbose = verbose
        cls._logdir = logdir
        cls._get_instance()

    def _start(self):
        now = datetime.now(tzlocal())
        self._accessid = _gen_hexid()
        self._uniqueid = "UNIQID"
        self._accesssec = now
        self._accesstime = now.strftime(Logger._dateformat)
        message = "LOG Start with ACCESSID=[%s] UNIQUEID=[%s] ACCESSTIME=[%s]"
        self._log(Logger.INFO, message % (self._accessid, self._uniqueid, self._accesstime))

    def _stop(self):
        processtime = '%3.9f' % (datetime.now(tzlocal()) - self._accesssec).total_seconds()
        message = "LOG End with ACCESSID=[%s] UNIQUEID=[%s] ACCESSTIME=[%s] PROCESSTIME=[%s]\n"
        self._log(Logger.INFO, message % (self._accessid, self._uniqueid, self._accesstime, processtime))
        while len(self._logger.handlers) > 0:
            self._remove_handler(self._logger.handlers[0])

    def _log(self, level, message, exc_info=False, stack_info=False):
        extras = {
            'accessid': self._accessid,
            'priority': Logger.LABELS[level]
        }
        self._logger.log(level, message, extra=extras, exc_info=exc_info, stack_info=stack_info)

    @classmethod
    def e(cls, message):
        cls._get_instance()._log(Logger.ERROR, message, exc_info=True, stack_info=True)

    @classmethod
    def w(cls, message):
        cls._get_instance()._log(Logger.WARNING, message)

    @classmethod
    def i(cls, message):
        cls._get_instance()._log(Logger.INFO, message)

    @classmethod
    def d(cls, message):
        cls._get_instance()._log(Logger.DEBUG, message)

    @classmethod
    def v(cls, message):
        cls._get_instance()._log(Logger.VERBOSE, message)


class App(Singleton, Callable):
    _basedir = os.path.dirname(os.path.realpath(__file__))
    _logdir = _basedir + '/logs'
    _loglevel = Logger.DEBUG
    _verbose = True
    _debug = True
    __defined_args = []

    @abstractmethod
    def main(self):
        raise NotImplementedError()

    @classmethod
    def configure(
            cls,
            logdir=_logdir,
            loglevel=_loglevel,
            verbose=_verbose,
            debug=_debug):
        cls._loglevel = loglevel
        cls._verbose = verbose
        cls._logdir = logdir
        cls._debug = debug

    @classmethod
    def exec(cls):
        app = cls()
        app()

    @classmethod
    def _def_arg(cls, *args, **kwargs):
        cls.__defined_args.append((args, kwargs))

    def __init__(self):
        self._basedir = os.path.dirname(os.path.realpath(getfile(self.__class__)))

    def _initialize(self):
        pass

    def __initialize(self):
        self._initialize()
        self._name = sys.argv[0]
        self._def_arg('--debug', type=str, default=self._debug,
                      help='Enable debug mode')
        self._def_arg('--logdir', type=str, default=self._logdir,
                      help='Log directory')
        self._def_arg('--silent', '--quiet', action='store_true', default=not(self._verbose),
                      help='Silent execution: does not print any message')
        parser = argparse.ArgumentParser()
        [parser.add_argument(*_args, **_kwargs) for (_args, _kwargs) in self.__defined_args]
        args = parser.parse_args()
        self.configure(args.logdir, Logger.DEBUG if args.debug else Logger.INFO, not(args.silent), args.debug)
        Logger.configure(loglevel=self._loglevel, verbose=self._verbose, logdir=self._logdir)
        if not self._verbose:
            sys.stdout = sys.stderr = open(os.devnull, 'w')
        self._args = args

    def __call__(self):
        self.__initialize()
        try:
            def handler(signum, frame):
                raise SystemExit("Signal(%d) received: The program %s will be closed" % (signum, __file__))
            signal.signal(signal.SIGINT, handler)
            signal.signal(signal.SIGTERM, handler)
            self.main()
        except Exception:
            Logger.e("Exception occurred during execution:")
        except SystemExit as e:
            Logger.w(e)
        finally:
            Logger.finalize()
            sys.exit(0)


if __name__ == "__main__":
    Log = Logger

    class MyApp(App):

        def _initialize(self):
            self._def_arg('--iter', '-i', type=int, default=10,
                          help='Number of iteration')

        def main(self):
            for i in range(self._args.iter):
                Log.i("Hello World!")

    MyApp.exec()
