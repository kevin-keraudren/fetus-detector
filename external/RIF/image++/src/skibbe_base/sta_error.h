/*#############################################################################
*
*	Copyright 2011 by Henrik Skibbe and Marco Reisert
* 
*     
*	This file is part of the STA-ImageAnalysisToolbox
* 
*	STA-ImageAnalysisToolbox is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
* 
*	STA-ImageAnalysisToolbox is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
* 
*	You should have received a copy of the GNU General Public License
*	along with STA-ImageAnalysisToolbox. 
*	If not, see <http://www.gnu.org/licenses/>.
*
*
*#############################################################################*/

#ifndef STA_COMMON_ERROR_H
#define STA_COMMON_ERROR_H
#include <ctime>

#include <list>
#include <string>
#include <iomanip>

#ifndef _WIN32
#include <sys/time.h>
#else
#include <windows.h>
#include < time.h >
#include <windows.h> //I've ommited this line.
#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif

struct timezone
{
	int  tz_minuteswest; /* minutes W of Greenwich */
	int  tz_dsttime;     /* type of dst correction */
};

int gettimeofday(struct timeval *tv, struct timezone *tz)
{
	FILETIME ft;
	unsigned __int64 tmpres = 0;
	static int tzflag;

	if (NULL != tv)
	{
		GetSystemTimeAsFileTime(&ft);

		tmpres |= ft.dwHighDateTime;
		tmpres <<= 32;
		tmpres |= ft.dwLowDateTime;

		/*converting file time to unix epoch*/
		tmpres -= DELTA_EPOCH_IN_MICROSECS;
		tmpres /= 10;  /*convert into microseconds*/
		tv->tv_sec = (long)(tmpres / 1000000UL);
		tv->tv_usec = (long)(tmpres % 1000000UL);
	}

	if (NULL != tz)
	{
		if (!tzflag)
		{
			_tzset();
			tzflag++;
		}
		tz->tz_minuteswest = _timezone / 60;
		tz->tz_dsttime = _daylight;
	}

	return 0;
}
#endif

namespace hanalysis
{

	/// the STA error class
	class STAError
	{
	public:
		STAError() {}

		STAError(const std::string& message)
			:_message(message)
		{}


		template<class T>
		STAError & operator<<(const T &data)
		{
			std::ostringstream os;
			os << data;
			_message += os.str();
			return *this;
		}

		/// \returns  the error string
		std::string str() const
		{
			return _message;
		}


		/// \returns  the error c-string
		const char* what() const
		{
			return _message.c_str();
		}

	private:
		std::string _message;

	};


	class CtimeStopper
	{
	private:
		double _startSecond;
		double _lastSecond;
		int steps;
		std::string startevent;
		std::list<std::string> event_names;
		std::list<double> event_times;
		std::list<double> event_all_times;
	public:
		CtimeStopper(std::string event)
		{
			startevent=event;
			steps=0;
			struct timeval _Time;
			gettimeofday(&_Time,NULL);
			_startSecond=_Time.tv_sec + 0.000001*_Time.tv_usec;
			_lastSecond=_startSecond;
		}
		~CtimeStopper()
		{
			if (startevent!="")
			{
				addEvent(startevent);
				printEvents();
			}
		}


		void addEvent(std::string event)
		{
			struct timeval  currentTime;
			gettimeofday(&currentTime,NULL);
			double currentSeconds=currentTime.tv_sec + 0.000001*currentTime.tv_usec;

			event_times.push_back(currentSeconds-_lastSecond);
			event_all_times.push_back(currentSeconds-_startSecond);
			event_names.push_back(event);	
			_lastSecond=currentSeconds;
			steps++;
		}    

		void printEvents(int precision=3)
		{
			printf("\n computation time (summary):\n");
			printf("    all         step\n");
			std::list<std::string>::iterator iter2=event_names.begin();
			std::list<double>::iterator iter3=event_all_times.begin();
			for (std::list<double>::iterator iter=event_times.begin();
				iter!=event_times.end();iter++,iter2++,iter3++)
			{
				std::stringstream s;
				s<<std::fixed<<std::setprecision(precision)<<*iter3<<" sec. | "<<*iter<<" sec. |\t("<<*iter2<<")";
				printf("%s\n",s.str().c_str());
			}
			printf("\n");
		}
	};


}

#endif

