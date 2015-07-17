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

#ifndef STA_FIELD_CLASS_H
#define STA_FIELD_CLASS_H

#include "stensor.h"
#include "sta_error.h"
#include "string.h"
#include "stensor_kernels.h"

#include <list>
#include <string>

namespace hanalysis
{
	static int classcount=0;
	static bool do_classcount=false;

	/// represents spherical tensor fields
	template<typename T>
	class _stafield
	{
	protected:
		int classcount_id;


		int object_is_dead_soon;

		// not fully supported by stensor.h yet 
		T element_size[3];

		/// image shape
		std::size_t shape[3];
		bool own_memory;

		/// must be either \ref  STA_FIELD_STORAGE_C, \ref STA_FIELD_STORAGE_R or \ref STA_FIELD_STORAGE_RF
		hanalysis::STA_FIELD_PROPERTIES field_storage;
		/// must be either \ref  STA_OFIELD_SINGLE, \ref STA_OFIELD_FULL, \ref STA_OFIELD_EVEN  or \ref STA_OFIELD_ODD
		hanalysis::STA_FIELD_PROPERTIES field_type;
		/// tensor rank \f$ \mathbf{stafield} \in \mathcal T_{L}\f$
		int L;

		// only relevant for views (selected components), else==0
		std::size_t stride;

		static void  set_death(const _stafield *  cfield)
		{
			_stafield * field=(_stafield* )  cfield;
			field->object_is_dead_soon++;
		}


	public:
		std::string name;

		std::size_t getNumVoxel() {
			return this->shape[0]*this->shape[1]*this->shape[2];
		}

		void switchFourierFlag() 
		{
			if (this->field_storage==STA_FIELD_STORAGE_R)
			{
				this->field_storage=STA_FIELD_STORAGE_RF;
				return;
			}
			if (this->field_storage==STA_FIELD_STORAGE_RF)
			{
				this->field_storage=STA_FIELD_STORAGE_R;
				return;
			}
		}



		static bool equalShape(const _stafield & a,const _stafield & b)
		{
			if  ((a.shape[0]!=b.shape[0]) ||
				(a.shape[1]!=b.shape[1]) ||
				(a.shape[2]!=b.shape[2]) )
				return false;
			return true;
		}    

		/*#############################################################
		*
		*     standard operators
		*
		*#############################################################*/

		/*!
		* returns true if \ref shape, \ref field_storage, \ref field_type and tensor rank \ref L  are equal, else false
		* */
		bool operator==(const _stafield & field) const
		{
			if  ((this->shape[0]!=field.shape[0]) ||
				(this->shape[1]!=field.shape[1]) ||
				(this->shape[2]!=field.shape[2]) ||
				(this->field_storage!=field.field_storage)||
				(this->field_type!=field.field_type)||
				(this->L!=field.L)
				)return false;
			return true;
		};

		/*!
		* returns false if \ref shape, \ref field_storage, \ref field_type and tensor rank \ref L  are equal, else false
		* */
		bool operator!=(const _stafield & field) const
		{
			return (!((*this)==field));
		};    



		/*#############################################################
		*
		*     constructors
		*
		*#############################################################*/    


		_stafield()
		{
			this->element_size[0]=T(1);
			this->element_size[1]=T(1);
			this->element_size[2]=T(1);
			this->field_storage=hanalysis::STA_FIELD_STORAGE_R;
			this->field_type=hanalysis::STA_OFIELD_SINGLE;
			this->shape[0]=0;
			this->shape[1]=0;
			this->shape[2]=0;
			//this->data=NULL;
			this->own_memory=true;
			this->L=-1;
			this->stride=0;
			classcount++;
			this->classcount_id=classcount;
			this->object_is_dead_soon=0;
		}    


		/*!
		* \returns constant pointer to the objects \ref shape structure
		* */
		const std::size_t *  getShape() const {
			return this->shape;
		};

		/*!
		* \returns \b true if objects has allocated its own memory, else \b false
		* */
		bool ownMemory() const {
			return this->own_memory;
		}

		bool oneBlockMem() const {
			return (this->stride==0);
		}    

		std::size_t  getStride() const {
			if (this->stride==0)
				return hanalysis::order2numComponents(this->field_storage,this->field_type,this->L);
			return this->stride;
		};

		void setElementSize(const T element_size[])
		{
			this->element_size[0]=element_size[0];
			this->element_size[1]=element_size[1];
			this->element_size[2]=element_size[2];
		}

		const T *  getElementSize() const
		{
			return this->element_size;
		}      


		hanalysis::STA_FIELD_PROPERTIES  getStorage() const
		{
			return this->field_storage;
		}

		hanalysis::STA_FIELD_PROPERTIES  getType() const
		{
			return this->field_type;
		}



		/*!
		\returns \f$ L \in \mathbb N \f$, the tensor rank
		*/
		int getRank() const {
			return this->L;
		}    


	};

	/// represents spherical tensor fields (CPU version)
	template<typename T>
	class stafield : public _stafield<T>
	{
	private:

		std::complex<T> * data;  
	public:

		/*#############################################################
		*
		*     standard operators
		*
		*#############################################################*/


		template<typename S>
		void operator=(S value)
		{
			if (this->getType()!=hanalysis::STA_OFIELD_SINGLE)
				throw  (hanalysis::STAError("operator= : field type must be STA_OFIELD_SINGLE"));

			int thestride=this->getStride();
			std::size_t jumpz=this->shape[1]*this->shape[2];
			int strid_all=jumpz*thestride;
			int numcomponents=hanalysis::order2numComponents(this->field_storage,this->field_type,this->L);
			int remaining=thestride-numcomponents;

#pragma omp parallel for num_threads(get_numCPUs())
			for (std::size_t z=0;z<this->shape[0];z++)
			{
				std::complex<T> * p=this->data+z*strid_all;
				for (std::size_t a=0;a<jumpz;a++)
				{
					for (int a=0;a<numcomponents;a++)
						(*p++)=value;
					p+=remaining;
				}
			}      
		}

		/*!
		*  \f$ \mathbf{stafield}:=\mathbf{f}, ~~ \mathbf{stafield},\mathbf{f} \in \mathcal T_{J}\f$  . Allocates new memory only when necessary
		* */
		stafield & operator=(const stafield & f)
		{

			if (this==&f)
				return *this;

			if ((this->L==-1)&&(f.L==-1))
				return *this;



			if ((f.object_is_dead_soon==1)
				&&(this->own_memory)
				)
			{



				if (f.object_is_dead_soon>1)
					throw STAError("error: something went wrong with the memory managemant \n");

				if (this->own_memory && (this->data!=NULL) && (this->object_is_dead_soon<2))
					delete [] this->data;

				this->set_death(&f);
				this->data=f.data;

				this->field_storage=f.field_storage;
				this->field_type=f.field_type;

				this->shape[0]=f.shape[0];
				this->shape[1]=f.shape[1];
				this->shape[2]=f.shape[2];

				this->setElementSize(f.getElementSize());

				this->L=f.L;

				this->stride=f.stride;
				this->own_memory=f.own_memory;
				return *this;
			}


			int check=0;

			if ((*this)!=f)
			{
				if (this->stride!=0)
					throw STAError("warning: operator=  (stride!=0) shared memory block but alocating new (own) memory would be nrequired \n");

				if (!this->own_memory)
					throw STAError("warning: operator=  (!own_memory): shared memory block but alocating new (own) memory would be nrequired \n");



				if (this->own_memory && (this->data!=NULL))
					delete [] this->data;

				this->field_storage=f.field_storage;
				this->field_type=f.field_type;
				this->shape[0]=f.shape[0];
				this->shape[1]=f.shape[1];
				this->shape[2]=f.shape[2];

				this->setElementSize(f.getElementSize());

				this->data=NULL;
				this->stride=0;
				this->own_memory=true;
				this->L=f.L;
				int numcomponents=hanalysis::order2numComponents(this->field_storage,this->field_type,this->L);
				int numVoxel=this->shape[0]*this->shape[1]*this->shape[2];
				this->data=new std::complex<T>[numcomponents*numVoxel];
				check=1;
			}

			if ((f.stride==0)&&(this->stride==0))
			{
				int numcomponents=hanalysis::order2numComponents(this->field_storage,this->field_type,this->L);
				this->setElementSize(f.getElementSize());
				memcpy(this->data,f.data,this->shape[0]*this->shape[1]*this->shape[2]*numcomponents*sizeof(std::complex<T>));
			}
			else
			{
				this->setElementSize(f.getElementSize());	  
				if (check==1)
					throw STAError("BullShit");

				if ((f.field_type!=hanalysis::STA_OFIELD_SINGLE)||
					(this->field_type!=hanalysis::STA_OFIELD_SINGLE))
					throw STAError("operator= this cannot happen ! the input field must be  STA_OFIELD_SINGLE");
				// view -> copy
				int numcomponents_new=this->L+1;
				if (this->field_storage==STA_FIELD_STORAGE_C)
					numcomponents_new=2*this->L+1;

				numcomponents_new*=2;
				int strid_in=2*(f.getStride())-numcomponents_new;
				int strid_out=2*this->getStride()-numcomponents_new;
				std::size_t jumpz=this->shape[1]*this->shape[2];
				int strid_in_all=jumpz*f.getStride();
				int strid_out_all=jumpz*this->getStride();

#pragma omp parallel for num_threads(get_numCPUs())
				for (std::size_t z=0;z<this->shape[0];z++)
				{
					T * in=(T*)(f.data+z*strid_in_all);
					T * out=(T*)(this->data+z*strid_out_all);
					for (std::size_t a=0;a<jumpz;a++)
					{
						for (int b=0;b<numcomponents_new;b++)
							(*out++)=(*in++);
						in+=strid_in;
						out+=strid_out;
					}
				}
			}

			return *this;
		}


		/*!
		*  \f$ \mathbf{stafield}:=\mathbf{stafield}+\mathbf{a}\f$ \n
		*  \f$  \mathbf{stafield},\mathbf{a} \in \mathcal T_{J}\f$
		* */
		stafield & operator+=(const stafield & a)
		{
			if (a!=*this)
				throw  (hanalysis::STAError("+=: shapes differ!"));
			try
			{
				Mult(a,*this);
			} catch (hanalysis::STAError & error)
			{
				throw error;
			}
			return *this;
		}

		/*!
		*  \f$ \mathbf{stafield}:=\mathbf{stafield}-\mathbf{a}\f$ \n
		* \f$  \mathbf{stafield},\mathbf{a} \in \mathcal T_{J}\f$
		* */
		stafield &  operator-=(const stafield & a)
		{
			if (a!=*this)
				throw  (hanalysis::STAError("-=: shapes differ!"));
			try
			{
				Mult(a,*this,-1);
			} catch (hanalysis::STAError & error)
			{
				throw error;
			}
			return *this;
		}

		/*!
		*  \f$ \mathbf{stafield}:=\alpha \mathbf{stafield} \f$ \n
		*  \f$  \mathbf{stafield} \in \mathcal T_{J}, \alpha \in  \mathbb C \f$
		* */
		stafield &  operator*=(std::complex<T> alpha)
		{
			try
			{
				Mult(*this,*this,alpha,false,true);
			} catch (hanalysis::STAError & error)
			{
				throw error;
			}
			return *this;
		}

		/*!
		*  \f$ \mathbf{stafield}:=\frac{\mathbf{stafield}}{\alpha} \f$ \n
		*  \f$  \mathbf{stafield} \in \mathcal T_{J}, \alpha \in  \mathbb C \f$
		* */
		stafield &  operator/=(std::complex<T> alpha)
		{
			try
			{
				Mult(*this,*this,T(1)/alpha,false,true);
			} catch (hanalysis::STAError & error)
			{
				throw error;
			}
			return *this;
		}

		/*!
		*  \f$ \mathbf{result}:=\mathbf{stafield}+\mathbf{a}\f$ \n
		*  \f$  \mathbf{stafield},\mathbf{a} \in \mathcal T_{J}\f$
		* */
		const stafield operator+(const stafield & a) const
		{
			if (a!=*this)
				throw  (hanalysis::STAError("-=: shapes differ!"));

			try
			{
				stafield result(this->shape,this->L,this->field_storage,this->field_type);
				Mult(*this,result,T(1),false,true);
				Mult(a,result,T(1),false,false);
				return  result;
			} catch (hanalysis::STAError & error)
			{
				throw error;
			}
		}


		/*!
		*  \f$ \mathbf{result}:=\mathbf{stafield}-\mathbf{a}\f$ \n
		*  \f$  \mathbf{stafield},\mathbf{a} \in \mathcal T_{J}\f$
		* */
		const stafield operator-(const stafield & a) const
		{
			if (a!=*this)
				throw  (hanalysis::STAError("-=: shapes differ!"));

			try
			{
				stafield result(this->shape,this->L,this->field_storage,this->field_type);
				Mult(*this,result,T(-1),false,true);
				Mult(a,result,T(-1),false,false);
				return  result;
			} catch (hanalysis::STAError & error)
			{
				throw error;
			}

		}


		/*!
		*  \f$ \mathbf{result}:=\mathbf{stafield}+\mathbf{a}\f$ \n
		*  \f$  \mathbf{stafield},\mathbf{a} \in \mathcal T_{J}\f$
		* */
		const stafield operator*(std::complex<T> alpha) const
		{
			try
			{
				stafield result(this->shape,this->L,this->field_storage,this->field_type);
				Mult(*this,result,alpha,false,true);
				return result;
			} catch (hanalysis::STAError & error)
			{
				throw error;
			}
		}

		/*!
		*  \f$ \mathbf{result}:=\mathbf{stafield}+\mathbf{a}\f$ \n
		*  \f$  \mathbf{stafield},\mathbf{a} \in \mathcal T_{J}\f$
		* */
		const stafield operator/(std::complex<T> alpha) const
		{
			try
			{
				stafield result(this->shape,this->L,this->field_storage,this->field_type);
				Mult(*this,result,T(1)/alpha,false,true);
				return result;
			} catch (hanalysis::STAError & error)
			{
				throw error;
			}

		}




		/*!
		\returns  a view on the (sub) component \f$ l \f$ with the   \n
		tensor rank  \f$ l \in \mathbb N \f$  of an orientation tensor field
		*/
		stafield operator[](int l) const
		{
			try
			{
				return this->get(l);
			}
			catch (hanalysis::STAError & error)
			{
				throw error;
			}
		}      


		/*#############################################################
		*
		*     constructors
		*
		*#############################################################*/



		stafield()  : _stafield<T>(), data(NULL) {};
		//     {
		//         this->element_size[0]=T(1);
		// 	this->element_size[1]=T(1);
		// 	this->element_size[2]=T(1);
		//         this->field_storage=hanalysis::STA_FIELD_STORAGE_R;
		//         this->field_type=hanalysis::STA_OFIELD_SINGLE;
		//         this->shape[0]=0;
		//         this->shape[1]=0;
		//         this->shape[2]=0;
		//         this->data=NULL;
		//         this->own_memory=true;
		//         this->L=-1;
		//         this->stride=0;
		//         classcount++;
		//         this->classcount_id=classcount;
		//         this->object_is_dead_soon=0;
		//     }

		stafield(const stafield & field) : _stafield<T>(), data(NULL)
		{
			//         this->element_size[0]=T(1);
			// 	this->element_size[1]=T(1);
			// 	this->element_size[2]=T(1);
			//         this->field_storage=hanalysis::STA_FIELD_STORAGE_R;
			//         this->field_type=hanalysis::STA_OFIELD_SINGLE;
			//         this->shape[0]=0;
			//         this->shape[1]=0;
			//         this->shape[2]=0;
			//         this->data=NULL;
			//         this->own_memory=true;
			// 	this->L=-1;
			//         this->stride=0;
			//         classcount++;
			//         this->classcount_id=classcount;
			//         this->object_is_dead_soon=0;

			(*this)=field;
		}





		/*! creates a spherical tensor field of order \f$ L \in \mathbb N \f$
		*   \param shape
		*  \param L \f$ L \in \mathbb N \f$, the tensor rank
		* */
		stafield(const std::size_t shape[],
			int L,
			hanalysis::STA_FIELD_PROPERTIES field_storage=hanalysis::STA_FIELD_STORAGE_R,
			hanalysis::STA_FIELD_PROPERTIES field_type=hanalysis::STA_OFIELD_SINGLE)
			: _stafield<T>(), data(NULL)
		{
			//         this->element_size[0]=T(1);
			// 	this->element_size[1]=T(1);
			// 	this->element_size[2]=T(1);      
			//         this->stride=0;
			if (L<0)
				throw STAError("L must be >= 0");
			this->field_storage=field_storage;
			this->field_type=field_type;
			this->shape[0]=shape[0];
			this->shape[1]=shape[1];
			this->shape[2]=shape[2];
			//         this->data=NULL;
			//         this->own_memory=true;
			this->L=L;
			int numcomponents=hanalysis::order2numComponents(field_storage,field_type,L);
			int numVoxel=shape[0]*shape[1]*shape[2];
			if (hanalysis::verbose>0)
				printf("L: %d , (%i,%i,%i) , // %i\n",L,shape[0],shape[1],shape[2],numcomponents);
			if (hanalysis::verbose>0)
				printf("allocating %i bytes\n",numcomponents*numVoxel*sizeof(std::complex<T>));
			this->data=new std::complex<T>[numcomponents*numVoxel];


			//         classcount++;
			//         this->classcount_id=classcount;
			//         this->object_is_dead_soon=0;
		}








		/*! creates a convolution kernel  of order \f$ L \in \mathbb N \f$
		* \param kernelname either "gauss","gaussLaguerre" or "gaussBessel"
		*   \param shape
		* \param params parameter vector [sigma],[sigma,n] or [sigma,k,s]
		* \param centered \f$ \in \{0,1\}  \f$
		*  \param L \f$ L \in \mathbb N \f$, the tensor rank
		* */
		stafield(std::string kernelname,
			const std::size_t shape[],
			std::vector<T> param_v,
			bool centered=false,
			int L=0,
			hanalysis::STA_FIELD_PROPERTIES field_storage=hanalysis::STA_FIELD_STORAGE_R)
			: _stafield<T>()
		{
			this->element_size[0]=T(1);
			this->element_size[1]=T(1);
			this->element_size[2]=T(1);        
			this->stride=0;
			if (L<0)
				throw STAError("L must be >= 0");
			this->field_storage=field_storage;
			this->field_type=hanalysis::STA_OFIELD_SINGLE;
			this->shape[0]=shape[0];
			this->shape[1]=shape[1];
			this->shape[2]=shape[2];
			this->data=NULL;
			this->own_memory=true;
			this->L=L;
			int numcomponents=hanalysis::order2numComponents(field_storage,this->field_type,L);
			int numVoxel=shape[0]*shape[1]*shape[2];
			if (hanalysis::verbose>0)
				printf("L: %d , (%i,%i,%i) , // %i\n",L,shape[0],shape[1],shape[2],numcomponents);
			if (hanalysis::verbose>0)
				printf("allocating %i bytes\n",numcomponents*numVoxel*sizeof(std::complex<T>));
			this->data=new std::complex<T>[numcomponents*numVoxel];

			makeKernel(kernelname,param_v,*this,centered);
			//         classcount++;
			//         this->classcount_id=classcount;
			//         this->object_is_dead_soon=0;
		}


		/*! creates a spherical tensor field of order \f$ L \in \mathbb N \f$ \n
		* based on existing data. No memory is allocated yet.
		*   \param shape
		*  \param L \f$ L \in \mathbb N \f$, the tensor rank
		* \param data pointer to existing memory
		* */
		stafield(const std::size_t shape[],
			int L,
			hanalysis::STA_FIELD_PROPERTIES field_storage,
			hanalysis::STA_FIELD_PROPERTIES field_type,
			std::complex<T>  * data) : _stafield<T>()
		{
			this->element_size[0]=T(1);
			this->element_size[1]=T(1);
			this->element_size[2]=T(1);        
			this->stride=0;
			if (L<0)
				throw  (hanalysis::STAError("L must be >= 0"));
			if (data==NULL)
				throw ( hanalysis::STAError("data is NULL-pointer"));
			this->field_storage=field_storage;
			this->field_type=field_type;
			this->shape[0]=shape[0];
			this->shape[1]=shape[1];
			this->shape[2]=shape[2];
			this->data=data;
			this->own_memory=false;
			this->L=L;
			//         classcount++;
			//         this->classcount_id=classcount;
			//         this->object_is_dead_soon=0;
		}






#ifdef _SUPPORT_MATLAB_
		stafield(mxArray  *  field,
			hanalysis::STA_FIELD_PROPERTIES field_storage,
			hanalysis::STA_FIELD_PROPERTIES field_type)
			: _stafield<T>()
		{
			stafieldFromMxArray(field,field_storage,field_type);
		}

		stafield(mxArray  *  field)  : _stafield<T>()
		{
			// 	if (isStaField<T>(field))
			// 	{
			// 	  mxArray *storage = mxGetPropertyPtr(field,0,(char *)"storage");
			// 	  mxArray *type = mxGetPropertyPtr(field,0,(char *)"type");
			// 	  mxArray *data = mxGetPropertyPtr(field,0,(char *)"data");
			// 	  stafieldFromMxArray(data,
			// 			      enumfromstring(mex2string(storage)),
			// 			      enumfromstring(mex2string(type)));
			// 	  return;
			// 	}
			if (isStaFieldStruct<T>(field))
			{
				mxArray *storage = mxGetField(field,0,(char *)"storage");
				mxArray *type = mxGetField(field,0,(char *)"type");
				mxArray *data = mxGetField(field,0,(char *)"data");
				stafieldFromMxArray(data,
					enumfromstring(mex2string(storage)),
					enumfromstring(mex2string(type)));
				return;
			}	
			throw ( hanalysis::STAError("mxArray contains no valid stafield class nor a valid starfield struct!"));
		}

	private:
		void stafieldFromMxArray(mxArray  *   field,
			hanalysis::STA_FIELD_PROPERTIES field_storage=hanalysis::STA_FIELD_STORAGE_R,
			hanalysis::STA_FIELD_PROPERTIES field_type=hanalysis::STA_OFIELD_SINGLE)
		{
			if (mxGetClassID(field)!=getclassid<T>())
				throw ( hanalysis::STAError("data type missmatch!"));

			this->element_size[0]=T(1);
			this->element_size[1]=T(1);
			this->element_size[2]=T(1);  
			this->stride=0;
			this->field_storage=field_storage;
			this->field_type=field_type;
			this->data=NULL;
			this->own_memory=false;
			const int *dimsFIELD = mxGetDimensions(field);
			const int numdimFIELD = mxGetNumberOfDimensions(field);
			if (numdimFIELD!=5)
				throw ( hanalysis::STAError("wrong number of dimensions"));
			if (dimsFIELD[0]!=2)
				throw ( hanalysis::STAError("wrong number of components in first dimension (must be 2 -> real/imag)"));
			int ncomponents=dimsFIELD[1];

			this->L=numComponents2order(field_storage,field_type,ncomponents);

			this->data = (std::complex<T> *) (mxGetData(field));
			this->shape[0]=dimsFIELD[2];
			this->shape[1]=dimsFIELD[3];
			this->shape[2]=dimsFIELD[4];
			std::swap(this->shape[0],this->shape[2]);
			if (hanalysis::verbose>0)
				printf("L: %d , (%i,%i,%i) , // %i\n",this->L,this->shape[0],this->shape[1],this->shape[2],ncomponents);
			//         classcount++;
			//         this->classcount_id=classcount;
			//         this->object_is_dead_soon=0;
		}

	public:


		//     static stafield  createFieldAndmxClass( mxArray  * & newMxArray,
		//                                             const std::size_t shape[],
		//                                             int L,
		//                                             hanalysis::STA_FIELD_PROPERTIES field_storage=hanalysis::STA_FIELD_STORAGE_R,
		//                                             hanalysis::STA_FIELD_PROPERTIES field_type=hanalysis::STA_OFIELD_SINGLE)
		//     {
		//         int numcomponents=hanalysis::order2numComponents(field_storage,field_type,L);
		// 
		//         int ndims[5];
		//         ndims[0] = 2;
		//         ndims[1] = numcomponents;
		//         ndims[2]=shape[2];
		//         ndims[3]=shape[1];
		//         ndims[4]=shape[0];
		// 
		//         mxArray *precision= mxCreateString(getPrecisionFlag<T>().c_str());
		//         mxArray *in[1];
		//         in[0] = precision;
		//         mxArray *out[1];
		//         mexCallMATLAB(1, out, 1, in , "stafield");
		//         newMxArray = out[0];
		//         int dim = 1;
		//         mxArray *theL = mxCreateNumericArray(1,&dim,getclassid<T>(),mxREAL);
		//         T *s =  (T*) mxGetData(theL);
		//         *s = L;
		//         int dimsh[] = {1,3};
		//         mxArray *theShape = mxCreateNumericArray(2,dimsh,getclassid<T>(),mxREAL);
		//         T *theshape =  (T*) mxGetData(theShape);
		//         theshape[0] = shape[2];
		//         theshape[1] = shape[1];
		//         theshape[2] = shape[0];
		// 
		//         mxSetProperty(newMxArray,0,"shape",theShape);
		//         mxSetProperty(newMxArray,0,"storage",mxCreateString(enumtostring(field_storage).c_str()));
		//         mxSetProperty(newMxArray,0,"type",mxCreateString(enumtostring(field_type).c_str()));
		//         mxSetProperty(newMxArray,0,"L",theL);
		//         mxSetProperty(newMxArray,0,"data",mxCreateNumericArray(5,ndims,getclassid<T>(),mxREAL));
		// //printf("getting pointer\n");
		// 	std::complex<T> * datap=(std::complex<T> *) (mxGetData(mxGetPropertyPtr(newMxArray,0,(char*)"data")));
		// //printf("done\n");	
		// 	stafield stOut(shape,L,field_storage,field_type,datap);
		//         set_death(&stOut);
		//         return stOut;
		//     }

		static stafield  createFieldAndmxStruct( mxArray  * & newMxArray,
			const std::size_t shape[],
			int L,
			hanalysis::STA_FIELD_PROPERTIES field_storage=hanalysis::STA_FIELD_STORAGE_R,
			hanalysis::STA_FIELD_PROPERTIES field_type=hanalysis::STA_OFIELD_SINGLE)
		{
			int numcomponents=hanalysis::order2numComponents(field_storage,field_type,L);

			int ndims[5];
			ndims[0] = 2;
			ndims[1] = numcomponents;
			ndims[2]=shape[2];
			ndims[3]=shape[1];
			ndims[4]=shape[0];

			const char *field_names[] = {"storage","type","L","data","shape"};

			int dim = 1;
			newMxArray=mxCreateStructArray(1,&dim,5,field_names);

			mxArray *theL = mxCreateNumericArray(1,&dim,mxDOUBLE_CLASS,mxREAL);
			double *s =  (double*) mxGetData(theL);
			*s = L;

			int dims[2]={1,3};
			dim=2;
			mxArray *theSHAPE = mxCreateNumericArray(dim,dims,mxDOUBLE_CLASS,mxREAL);
			s =  (double*) mxGetData(theSHAPE);
			s[0] = shape[2];
			s[1] = shape[1];
			s[2] = shape[0];


			mxSetField(newMxArray,0,"storage", mxCreateString(enumtostring(field_storage).c_str()));	
			mxSetField(newMxArray,0,"type", mxCreateString(enumtostring(field_type).c_str()));
			mxSetField(newMxArray,0,"L", theL);
			mxSetField(newMxArray,0,"data", mxCreateNumericArray(5,ndims,getclassid<T>(),mxREAL));
			mxSetField(newMxArray,0,"shape", theSHAPE);	


			std::complex<T> * datap=(std::complex<T> *) (mxGetData(mxGetField(newMxArray,0,(char *)"data")));

			stafield stOut(shape,L,field_storage,field_type,datap);
			set_death(&stOut);
			return stOut;
		}


		static stafield  createFieldAndmxArray( mxArray  * & newMxArray,
			const std::size_t shape[],
			int L,
			hanalysis::STA_FIELD_PROPERTIES field_storage=hanalysis::STA_FIELD_STORAGE_R,
			hanalysis::STA_FIELD_PROPERTIES field_type=hanalysis::STA_OFIELD_SINGLE)
		{
			int numcomponents=hanalysis::order2numComponents(field_storage,field_type,L);

			int ndims[5];
			ndims[0] = 2;
			ndims[1] = numcomponents;
			ndims[2]=shape[2];
			ndims[3]=shape[1];
			ndims[4]=shape[0];
			newMxArray = mxCreateNumericArray(5,ndims,getclassid<T>(),mxREAL);

			stafield stOut(shape,L,field_storage,field_type, (std::complex<T> *) (mxGetData(newMxArray)));
			set_death(&stOut);
			return stOut;
		}

#endif





		~stafield()
		{
			if (this->own_memory && (this->data!=NULL))
			{
				if (do_classcount)
					printf("destroying stafield %i / remaining:  %i  [own]",this->classcount_id,--classcount);

				if (this->object_is_dead_soon<2)
				{
					if (do_classcount)
						printf(" (deleting data)\n");

					if (hanalysis::verbose>0)
						printf("field destrucor -> deleting data\n");
					delete [] this->data;
				}
				else
					if (do_classcount)
						printf(" (not deleting data, still having references)\n");
			} else
			{
				if (do_classcount)
				{
					printf("destroying stafield %i / remaining:  %i  [view]\n",this->classcount_id,--classcount);
				}
				if (hanalysis::verbose>0)
					printf("field destrucor -> --\n");
			}


		}


		/*#############################################################
		*
		*     methods
		*
		*#############################################################*/


		/*!
		* ensures that the objects has allocated its own memory
		* \returns \b false if objects has already allocated its own memory
		* */
		bool createMemCopy()
		{
			if  (this->own_memory) return false;

			// 	 if (this->field_type!=hanalysis::STA_OFIELD_SINGLE)
			//                 throw STAError("operator= this cannot happen ! the input field must be  STA_OFIELD_SINGLE");

			std::complex<T> * own_mem=NULL;

			// copy here
			if (this->stride==0)
			{
				int numcomponents=hanalysis::order2numComponents(this->field_storage,this->field_type,this->L);
				if (hanalysis::verbose>0)
					printf("copying field with %i components\n",numcomponents);
				if (hanalysis::verbose>0)
					printf("copying %i bytes\n",this->shape[0]*this->shape[1]*this->shape[2]*numcomponents*sizeof(std::complex<T>));
				own_mem=new std::complex<T>[this->shape[0]*this->shape[1]*this->shape[2]*numcomponents];
				memcpy(own_mem,this->data,this->shape[0]*this->shape[1]*this->shape[2]*numcomponents*sizeof(std::complex<T>));
			}
			else
			{
				int numcomponents=hanalysis::order2numComponents(this->field_storage,this->field_type,this->L);
				own_mem=new std::complex<T>[this->shape[0]*this->shape[1]*this->shape[2]*numcomponents];

				//printf("%d %d\n",numcomponents,this->getStride());
				numcomponents*=2;

				int strid_in=2*(this->getStride())-numcomponents;
				std::size_t jumpz=this->shape[1]*this->shape[2];
				int strid_in_all=jumpz*this->getStride();
				int strid_out_all=jumpz*numcomponents/2;

#pragma omp parallel for num_threads(get_numCPUs())
				for (std::size_t z=0;z<this->shape[0];z++)
				{
					T * in=(T*)(this->data+z*strid_in_all);
					T * out=(T*)(own_mem+z*strid_out_all);
					for (std::size_t a=0;a<jumpz;a++)
					{
						for (int b=0;b<numcomponents;b++)
							(*out++)=(*in++);
						in+=strid_in;
					}
				}
				this->stride=0;	    
			}

			std::swap(this->data,own_mem);
			this->own_memory=true;
			this->object_is_dead_soon=0;
			return true;
		}


		/*!
		* \returns data pointer
		* */
		std::complex<T> *  getData() {
			return this->data;
		};

		/*!
		* \returns constant data pointer
		* */
		const std::complex<T> *  getDataConst() const {
			return this->data;
		};





		/*!
		\returns  a view on the (sub) component \f$ l \f$ with the   \n
		tensor rank  \f$ l \in \mathbb N \f$  of an orientation tensor field
		*/
		stafield get(int l) const
		{
			if (l<0)
				throw  (hanalysis::STAError("error retrieving (sub) field l, l must be >= 0"));
			if (l>this->L)
				throw  (hanalysis::STAError("error retrieving (sub) field l, l must be <= L"));

			if ((this->field_type==STA_OFIELD_ODD)&&(l%2==0))
				throw  (hanalysis::STAError("error retrieving (sub) field l, l must be odd"));

			if ((this->field_type==STA_OFIELD_EVEN)&&(l%2!=0))
				throw  (hanalysis::STAError("error retrieving (sub) field l, l must be even"));



			std::complex<T> * component_data;
			int offset=hanalysis::getComponentOffset(this->field_storage,this->field_type,l);
			int numcomponents=hanalysis::order2numComponents(this->field_storage,this->field_type,this->L);

			component_data=this->data+offset;

			stafield  view(this->shape,
				l,
				this->field_storage,
				hanalysis::STA_OFIELD_SINGLE,
				component_data);
			this->set_death(&view);
			view.stride=numcomponents;

			view.setElementSize(this->getElementSize());
			return view;
		}







		/*#############################################################
		*
		*     static STA operators
		*
		*#############################################################*/

		/// tensor fft component by component
		/*!
		transforms  a spherical tensor field \f$ \mathbf{stIn} \in \mathcal T_{J}\f$ into Fourier domain (and back) \n
		\param stIn \f$ \mathbf{stIn} \in \mathcal T_{J}\f$
		\param stOut \f$ \mathbf{stOut} \in \mathcal T_{J}\f$
		\param forward
		\f$
		\left\{
		\begin{array}{ll}
		\mathbf{stOut}=\alpha\mathcal F (\mathbf{stIn}) &  \mbox{ if } forward=true \\
		\mathbf{stOut}=\alpha\mathcal F^{-1} (\mathbf{stIn}) &  \mbox{ if } forward=false
		\end{array}
		\right.
		\f$
		\param conjugate if \b true it computes \f$ \alpha\overline{\mathcal F (\mathbf{stIn})} \f$ and   \f$ \alpha\overline{\mathcal F^{-1} (\mathbf{stIn})} \f$, respectively
		\param alpha \f$ \alpha \in \mathbb C \f$ additional weighting factor
		\warning
		Consider that  \f$   \frac{\mathcal F^{-1}(\mathcal F (\mathbf{stIn}))}{shape[0] \cdot shape[1] \cdot shape[2]}=\mathbf{stIn}   \f$  !!
		*/
		static void FFT(const stafield & stIn,
			stafield & stOut,
			bool forward,
			bool conjugate=false,
			std::complex<T> alpha= T( 1 ),
#ifdef _STA_LINK_FFTW	       
			int flag=FFTW_ESTIMATE )
#else
			int flag=0 )
#endif
		{
			if ((!stafield::equalShape(stIn,stOut))||
				(stIn.field_type!=stOut.field_type)||
				(stIn.L!=stOut.L)||
				(((stIn.field_storage==STA_FIELD_STORAGE_C)||
				(stOut.field_storage==STA_FIELD_STORAGE_C))&&
				(stIn.field_storage!=stOut.field_storage)))
				// 		  (!((((stIn.field_storage==hanalysis::STA_FIELD_STORAGE_R)
				// 		  ||(stIn.field_storage==hanalysis::STA_FIELD_STORAGE_RF))&&
				// 		    ((stOut.field_storage==hanalysis::STA_FIELD_STORAGE_R)||
				// 		    (stOut.field_storage==hanalysis::STA_FIELD_STORAGE_RF)))||
				// 		    (stIn.field_storage==stOut.field_storage))))
				throw  (hanalysis::STAError("FFT: shapes differ!"));

			std::size_t ncomponents_in=hanalysis::order2numComponents(stIn.getStorage(),stIn.getType(),stIn.L);
			std::size_t ncomponents_out=hanalysis::order2numComponents(stOut.getStorage(),stOut.getType(),stOut.L);    
			if (((stIn.stride!=0)&&(ncomponents_in!=stIn.stride))||((stOut.stride!=0)&&(ncomponents_out!=stOut.stride)))
				throw  (hanalysis::STAError("FFT: doesn't work on this kind of views!"));
			if ((stIn.data==stOut.data))
				throw  (hanalysis::STAError("FFT: inplace not supported!"));


			hanalysis::STA_FIELD_PROPERTIES new_field_storage=stIn.getStorage();
			if (new_field_storage==hanalysis::STA_FIELD_STORAGE_R)
				new_field_storage=hanalysis::STA_FIELD_STORAGE_RF;
			else
				if (new_field_storage==hanalysis::STA_FIELD_STORAGE_RF)
					new_field_storage=hanalysis::STA_FIELD_STORAGE_R;
			stOut.field_storage=new_field_storage;

			if ((stIn.getStorage()==STA_FIELD_STORAGE_C)&&(stOut.getStorage()!=STA_FIELD_STORAGE_C))
				throw  (hanalysis::STAError("FFT: storage type must be the same"));




			int  stride_in = stIn.getStride();
			int  stride_out = stOut.getStride();
			int ncomponents=hanalysis::order2numComponents(stIn.getStorage(),stIn.getType(),stIn.L);

			if (hanalysis::verbose>0)
				printf("FFT: stride_in: %i stride_out %i , ncomp: %i\n",stride_in,stride_out,ncomponents);
			//int err=0;
			hanalysis::sta_fft(
				stIn.getDataConst(),
				stOut.getData(),
				stIn.getShape(),
				ncomponents,
				forward,
				conjugate,
				alpha,
				flag);
		}




		/// computes \f$ \alpha(\mathbf{stIn})  \f$
		/*!
		multiplication with a scalar: \f$ \alpha(\mathbf{stIn})  \f$ \n
		\param stIn \f$ \mathbf{stIn1} \in \mathcal T_{J}\f$
		\param stOut \f$ \alpha(\mathbf{stIn}) \in \mathcal T_{J} \f$
		\param alpha \f$ \alpha \in \mathbb C \f$  weighting factor
		\param conjugate returns \f$ \alpha(\overline{\mathbf{stIn}})\f$ if \b true
		*/
		static void Mult(const stafield & stIn,
			stafield & stOut,
			std::complex<T> alpha= T( 1 ),
			bool conjugate=false,
			bool clear_result = false)
		{
			if (stIn!=stOut)
				throw  (hanalysis::STAError("Mult: shapes differ!"));
			if (stIn.getStorage()!=stOut.getStorage())
				throw  (hanalysis::STAError("Mult: storage type must be the same"));
			int  stride_in = stIn.getStride();
			int  stride_out = stOut.getStride();
			int ncomponents=hanalysis::order2numComponents(stIn.getStorage(),stIn.getType(),stIn.L);

			if (hanalysis::verbose>0)
				printf("Mult: stride_in: %i stride_out %i , ncomp: %i\n",stride_in,stride_out,ncomponents);

			int err=hanalysis::sta_mult<T,std::complex<T> > (
				stIn.getDataConst(),
				stOut.getData(),
				stIn.getShape(),
				ncomponents,
				alpha,
				conjugate,
				stride_in,
				stride_out,
				clear_result);

			if (err==-1)
				throw  (hanalysis::STAError("Mult: error!"));

		}




		/// returns lengths of vectors component by compnent \n
		/*!
		\param stIn \f$ \mathbf{stIn1} \in \mathcal T_{J}\f$
		\param stOut \f$ \alpha(\mathbf{stIn}) \in \mathcal T_{0} \f$
		*/
		static void Norm(const stafield & stIn,
			stafield & stOut,
			bool clear_result = false)
		{
			if (!stafield::equalShape(stIn,stOut))
				throw  (hanalysis::STAError("Norm: shapes differ!"));
			if (stOut.getRank()!=0)
				throw  (hanalysis::STAError("Norm: stOut must be 0-order tensor field!!"));
			if (stIn.getType()!=hanalysis::STA_OFIELD_SINGLE)
				throw  (hanalysis::STAError("Deriv: first input field type must be STA_OFIELD_SINGLE"));
			if (stOut.getType()!=stIn.getType())
				throw  (hanalysis::STAError("Deriv: stOut field type must be STA_OFIELD_SINGLE"));

			if (stIn.getStorage()!=stOut.getStorage())
				throw  (hanalysis::STAError("Norm: storage type must be the same"));
			int  stride_in = stIn.getStride();
			int  stride_out = stOut.getStride();

			if (hanalysis::verbose>0)
				printf("Norm: stride_in: %i stride_out %i\n",stride_in,stride_out);

			int err=hanalysis::sta_norm<T> (
				stIn.getDataConst(),
				stOut.getData(),
				stIn.getShape(),
				stIn.getRank(),
				stIn.getStorage(),
				stride_in,
				stride_out,
				clear_result);

			if (err==-1)
				throw  (hanalysis::STAError("Norm: error!"));

		}


		/// spherical tensor derivative: \f$ \alpha( {\nabla}  \bullet_{(J+b)}  \mathbf{stIn}) , b \in \{-1,0,1\} \f$
		/*!
		computes the spherical tensor derivative of  \f$ \mathbf{stIn} \in \mathcal T_{J}\f$ \n
		\param stIn \f$ \mathbf{stIn} \in \mathcal T_{J}\f$
		\param stOut \f$ \mathbf{stOut} \in \mathcal T_{(J+Jupdown)}\f$, the spherical tensor derivative of \f$ \mathbf{stIn} \f$
		\param Jupdown
		\f$
		\left\{
		\begin{array}{ll}
		\mathbf{stOut}=\alpha({\nabla}  \bullet_{(J+1)}  \mathbf{stIn}), &  \mbox{ if } Jupdown=1\\
		\mathbf{stOut}=\alpha({\nabla}  \circ_{J}  \mathbf{stIn}), &  \mbox{ if } Jupdown=0\\
		\mathbf{stOut}=\alpha({\nabla}  \bullet_{(J-1)}  \mathbf{stIn}), &  \mbox{ if } Jupdown=-1
		\end{array}
		\right.
		\f$
		\param conjugate  if \b conjugate=true the  conjugate operator \f$ \overline{{\nabla}} \f$ is used
		\param alpha \f$ \alpha \in \mathbb C \f$ additional weighting factor
		\returns  \f$
		\left\{
		\begin{array}{ll}
		J+Jupdown &  \mbox{if derivative exists}\\
		-1 & \mbox{ else }
		\end{array}
		\right.
		\f$
		\warning if not \ref STA_FIELD_STORAGE_C then \b alpha must be real valued
		*/
		static void Deriv(const stafield & stIn,
			stafield & stOut,
			int Jupdown,
			bool conjugate=false,
			std::complex<T> alpha= T( 1 ),
			bool clear_result = false,
			int accuracy=0
			)
		{
			if (!stafield::equalShape(stIn,stOut))
				throw  (hanalysis::STAError("Deriv: shapes differ!"));
			if (stIn.getStorage()!=stOut.getStorage())
				throw  (hanalysis::STAError("Deriv: storage type must be the same"));
			if (stIn.getType()!=hanalysis::STA_OFIELD_SINGLE)
				throw  (hanalysis::STAError("Deriv: first input field type must be STA_OFIELD_SINGLE"));
			if (stOut.getType()!=stIn.getType())
				throw  (hanalysis::STAError("Deriv: stOut field type must be STA_OFIELD_SINGLE"));
			if (stOut.getRank()!=stIn.getRank()+Jupdown)
				throw  (hanalysis::STAError("Deriv: stOut rank must be input rank+Jupdown"));

			int  stride_in = stIn.getStride();
			int  stride_out = stOut.getStride();

			if (hanalysis::verbose>0)
				printf("Deriv: stride_in: %i stride_out %i\n",stride_in,stride_out);

			int err=hanalysis::sta_derivatives(
				stIn.getDataConst(),
				stOut.getData(),
				stIn.getShape(),
				stIn.getRank(),
				Jupdown,
				conjugate,
				alpha,
				stIn.getStorage(),
				stIn.getElementSize(),
				stride_in,
				stride_out,
				clear_result,
				accuracy);

			if (err==-1)
				throw  (hanalysis::STAError("Deriv: error!"));


		}






		/// Laplacian: \f$ \alpha\triangle(\mathbf{stIn}) \in \mathcal T_{J}\f$
		/*!
		* computes the Laplacian of \f$ \mathbf{stIn} \in \mathcal T_{J}\f$ component by component \n
		* \param stIn \f$ \mathbf{stIn} \in \mathcal T_{J}\f$
		* \param stOut \f$ \alpha\triangle(\mathbf{stIn}) \in \mathcal T_{J}\f$
		* \param shape
		* \param components \f$ components \in \mathbb N_{>0} \f$ number of tensor components of the input field \f$ \mathbf{stIn}  \f$
		* \param type if \b type=1 the standard 6 neighbours operator is used, if \b type=0 a 18 neighbours operator is used
		*  \param alpha \f$ \alpha \in \mathbb C \f$ additional weighting factor
		*
		*/
		static void Lap(const stafield & stIn,
			stafield & stOut,
			std::complex<T> alpha= T( 1 ),
			bool clear_result = false,
			int type=1)
		{
			if (stIn.getStorage()!=stOut.getStorage())
				throw  (hanalysis::STAError("Lap: storage type must be the same"));      
			if (stIn!=stOut)
				throw  (hanalysis::STAError("Lap: shapes differ!"));
			int  stride_in = stIn.getStride();
			int  stride_out = stOut.getStride();
			int ncomponents=hanalysis::order2numComponents(stIn.getStorage(),stIn.getType(),stIn.L);


			if (hanalysis::verbose>0)
				printf("Lap: stride_in: %i stride_out %i , ncomp: %i\n",stride_in,stride_out,ncomponents);

			int err=hanalysis::sta_laplace (
				stIn.getDataConst(),
				stOut.getData(),
				stIn.getShape(),
				ncomponents,
				type,
				alpha,
				stIn.getStorage(),
				stIn.getElementSize(),//(T * )NULL,
				stride_in,
				stride_out,
				clear_result);

			if (err==-1)
				throw  (hanalysis::STAError("Lap: error!"));


		}

		///  spherical tensor product:  \f$ \alpha(\mathbf{stIn1} \circ_{J} \mathbf{stIn2}) \f$ and \f$ \alpha(\mathbf{stIn1} \bullet_{J} \mathbf{stIn2}) \f$, respectively  \n
		/*!
		computes the spherical tensor product \f$ \alpha(\mathbf{stIn1} \circ_{J} \mathbf{stIn2}) \f$ and \f$ \alpha(\mathbf{stIn1} \bullet_{J} \mathbf{stIn2}) \f$, respectively  \n
		\param stIn1 \f$ \mathbf{stIn1} \in \mathcal T_{J_1}\f$
		\param stIn2 \f$ \mathbf{stIn2} \in \mathcal T_{J_2} \f$
		\param stOut \f$ \alpha(\mathbf{stIn1} \bullet_{J} \mathbf{stIn2}) \in \mathcal T_{J}\f$ if normalized, \f$ \alpha(\mathbf{stIn1} \circ_{J} \mathbf{stIn2}) \in \mathcal T_{J}\f$  else
		\param J \f$ J \in \mathbb N \f$ tensor rank of the resulting field
		\param normalize normalized tensor products?: true=\f$ \bullet_{J}\f$ , false=\f$ \circ_{J}\f$
		\param alpha \f$ \alpha \in \mathbb C \f$ additional weighting factor    
		\returns  \f$
		\left\{
		\begin{array}{ll}
		0 &  \mbox{if tensor product exists}\\
		-1 & \mbox{ else }
		\end{array}
		\right.
		\f$
		\warning If \b field_property= \ref STA_FIELD_STORAGE_R and  \f$ (J_1+J2+J)\%2 \neq 0 \f$ the
		function returns \n \f$ (\mathit{i})\alpha(\mathbf{stIn1} \circ_{J} \mathbf{stIn2}) \in \mathcal T_{J}\f$.
		This ensures that \ref STA_FIELD_STORAGE_R holds for \f$ \mathbf{stOut} \f$, too. \n
		The same is true for \b field_property=\ref STA_FIELD_STORAGE_RF
		\warning if not \ref STA_FIELD_STORAGE_C then \b alpha must be real valued
		*/
		static void Prod(const stafield & stIn1,
			const stafield & stIn2,
			stafield & stOut,
			int J,
			bool normalize=false,		     
			std::complex<T> alpha= T( 1 ),
			bool clear_result = false)
		{
			// 	printf("storage stIn1: %s \n",enumtostring(stIn1.getStorage()).c_str());
			// 	printf("storage stIn2: %s \n",enumtostring(stIn2.getStorage()).c_str());
			// 	printf("storage stOut: %s \n",enumtostring(stOut.getStorage()).c_str());

			//printf("%d %d %d \n",stIn1.getRank(),stIn2.getRank(),stOut.getRank());

			if ( ( std::abs ( stIn1.getRank()-stIn2.getRank() ) >J ) || 
				( J>std::abs ( stIn1.getRank()+stIn2.getRank() ) ) )
				throw  (hanalysis::STAError("Prod: ensure that |l1-l2|<=J && |l1+l2|<=J"));
			if ( ( ( stIn1.getRank()+stIn2.getRank()+J ) %2!=0 ) && ( normalize ) )
				throw  (hanalysis::STAError("Prod: ensure that l1+l2+J even"));    

			if (stOut.getRank()!=J)
				throw  (hanalysis::STAError("Prod: ensure that stOut has wrong Rank!"));    


			if ((!stafield::equalShape(stIn1,stOut))||(!stafield::equalShape(stIn2,stOut)))
				throw  (hanalysis::STAError("Prod: shapes differ!"));
			if ((stIn1.getStorage()!=stOut.getStorage())||(stIn2.getStorage()!=stOut.getStorage()))
				throw  (hanalysis::STAError("Prod: storage type must be the same"));
			if (stIn1.getType()!=hanalysis::STA_OFIELD_SINGLE)
				throw  (hanalysis::STAError("Prod: first input field type must be STA_OFIELD_SINGLE"));
			if (stIn1.getType()!=stIn2.getType())
				throw  (hanalysis::STAError("Prod: first input field type must be STA_OFIELD_SINGLE"));
			if (stOut.getType()!=stIn1.getType())
				throw  (hanalysis::STAError("Prod: stOut field type must be STA_OFIELD_SINGLE"));

			int  stride_in1 = stIn1.getStride();
			int  stride_in2 = stIn2.getStride();
			int  stride_out = stOut.getStride();

			if (hanalysis::verbose>0)
				printf("Prod: stride_in1: %i,stride_in2: %i stride_out %i\n",stride_in1,stride_in2,stride_out);




			int err=hanalysis::sta_product(
				stIn1.getDataConst(),
				stIn2.getDataConst(),
				stOut.getData(),
				stIn1.getShape(),
				stIn1.getRank(),
				stIn2.getRank(),
				stOut.getRank(),
				alpha,
				normalize,
				stIn1.getStorage(),
				stride_in1,
				stride_in2,
				stride_out,
				clear_result);

			if (err==-1)
			{
				throw  (hanalysis::STAError("Prod: error!"));
			}

		}



		static std::vector<T> kernel_param(std::string params)
		{
			for (std::size_t a=0;a<params.length();a++)
				if (params[a]==',')
					params[a]=' ';
			std::vector<T> param_v;
			std::stringstream param_s;
			param_s<<params;
			int eexit=0;
			while (!param_s.eof() && param_s.good())
			{
				T tmp;
				param_s>>tmp;
				param_v.push_back(tmp);
				eexit++;
				if (eexit>42)
					throw STAError("ahhh, something went completely wrong while parsing parameters! ");
			}
			return param_v;
		}


		static void makeKernel(std::string kernelname,
			std::vector<T> param_v,
			stafield & stIn,
			bool centered=false
			)
		{
			if (stIn.getType()!=hanalysis::STA_OFIELD_SINGLE)
				throw  (hanalysis::STAError("makeKernel: first input field type must be STA_OFIELD_SINGLE"));

			hanalysis::Kernel<T> * currentKernel=NULL;
			hanalysis::STA_CONVOLUTION_KERNELS kernel=hanalysis::STA_CONV_KERNEL_UNKNOWN;

			if (kernelname=="gauss")
				kernel=hanalysis::STA_CONV_KERNEL_GAUSS;
			if (kernelname=="gaussLaguerre")
				kernel=hanalysis::STA_CONV_KERNEL_GAUSS_LAGUERRE;
			if (kernelname=="gaussBessel")
				kernel=hanalysis::STA_CONV_KERNEL_GAUSS_BESSEL;


			switch (kernel)
			{

			case hanalysis::STA_CONV_KERNEL_GAUSS:
				{
					if (!(param_v.size()==1))
						throw STAError("error: wrong numger of parameters");

					currentKernel=new hanalysis::Gauss<T>();
					((hanalysis::Gauss<T> *)currentKernel)->setSigma(param_v[0]);
				}break;
			case hanalysis::STA_CONV_KERNEL_GAUSS_LAGUERRE:
				{
					if (!(param_v.size()==2))
						throw STAError("error: wrong numger of parameters");
					currentKernel=new hanalysis::GaussLaguerre<T>();
					((hanalysis::GaussLaguerre<T> *)currentKernel)->setSigma(param_v[0]);
					((hanalysis::GaussLaguerre<T> *)currentKernel)->setDegree(std::floor(param_v[1]));

				}break;
			case hanalysis::STA_CONV_KERNEL_GAUSS_BESSEL:
				{
					if (!(param_v.size()==3))
						throw STAError("error: wrong numger of parameters");
					currentKernel=new hanalysis::GaussBessel<T>();
					((hanalysis::GaussBessel<T> *)currentKernel)->setSigma(param_v[0]);
					((hanalysis::GaussBessel<T> *)currentKernel)->setFreq(param_v[1]);
					((hanalysis::GaussBessel<T> *)currentKernel)->setGauss(param_v[2]);

				}break;
			default :
				throw STAError("error: unsupported kernel  \n");
			}

			int range=0;
			if (stIn.getStorage()==hanalysis::STA_FIELD_STORAGE_C)
				range=stIn.getRank();

			//T * v_size=NULL;

			for (int m=-stIn.getRank();m<=range;m++)
			{
				hanalysis::renderKernel(
					stIn.getData(),
					stIn.getShape(),
					currentKernel,
					stIn.getRank(),
					m,
					centered,
					stIn.getStorage(),
					stIn.getElementSize(),//v_size,
					stIn.getStride());
			}

			delete currentKernel;
		}

		/*#############################################################
		*
		*     STA operators
		*
		*#############################################################*/


		///  see \ref FFT
		stafield fft(bool forward,
			bool conjugate=false,
			std::complex<T> alpha= T( 1 ),
#ifdef _STA_LINK_FFTW	       
			int flag=FFTW_ESTIMATE )
#else
			int flag=0 ) const
#endif
		{
			try
			{
				stafield result(this->shape,this->L,this->field_storage,this->field_type);
				this->set_death(&result);
				FFT(*this,result,forward,conjugate,alpha,flag);
				return  result;
			}
			catch (hanalysis::STAError & error)
			{
				throw error;
			}
		}

		///  see \ref FFT
		stafield convolve(stafield & b,
			int J=0,
#ifdef _STA_LINK_FFTW	       
			int flag=FFTW_ESTIMATE )
#else
			int flag=0 )
#endif
		{
			try
			{
				return (*this).fft(true,false,T(1),flag).prod(b.fft(true,true,T(1),flag),J,true).fft(false,false,T(1)/T(b.getNumVoxel()),flag);;
			}
			catch (hanalysis::STAError & error)
			{
				throw error;
			}
		}



		///  see \ref Mult
		stafield  mult(std::complex<T> alpha= T( 1 ),
			bool conjugate=false) const
		{

			try
			{
				stafield result(this->shape,this->L,this->field_storage,this->field_type);
				this->set_death(&result);
				Mult(*this,result,alpha,conjugate,true);
				return  result;
			}
			catch (hanalysis::STAError & error)
			{
				throw error;
			}
		}






		///   see \ref Norm
		stafield  norm() const
		{

			try
			{
				stafield result(this->shape,0,this->field_storage,this->field_type);

				this->set_death(&result);
				Norm(*this,result,true);
				return  result;
			}
			catch (hanalysis::STAError & error)
			{
				throw error;
			}
		}


		///  see \ref Deriv
		stafield  deriv(int J,
			bool conjugate=false,
			std::complex<T> alpha= T( 1 ),
			int accuracy=0
			) const
		{
			try
			{
				stafield result(this->shape,this->L+J,this->field_storage,this->field_type);

				this->set_death(&result);
				Deriv(*this,result,J,conjugate,alpha,true,accuracy);
				return  result;
			}
			catch (hanalysis::STAError & error)
			{
				throw error;
			}

		}


		/// see \ref Deriv2
		stafield  deriv2(int J,
			bool conjugate=false,
			std::complex<T> alpha= T( 1 ),
			int accuracy=0
			) const
		{

			try
			{
				stafield result(this->shape,this->L+J,this->field_storage,this->field_type);
				this->set_death(&result);
				Deriv2(*this,result,J,conjugate,alpha,true);
				return  result;
			}
			catch (hanalysis::STAError & error)
			{
				throw error;
			}
		}

		/// see \ref Lap
		stafield    lap(std::complex<T> alpha= T( 1 ),
			int type=1) const
		{

			try
			{
				stafield result(this->shape,this->L,this->field_storage,this->field_type);
				this->set_death(&result);
				Lap(*this,result,alpha,true,type);
				return  result;
			}
			catch (hanalysis::STAError & error)
			{
				throw error;
			}
		}




		///  see \ref Prod
		stafield  prod(const stafield & b,
			int J,
			bool normalize=false,
			std::complex<T> alpha= T( 1 )) const
		{
			try
			{
				stafield result(this->shape,J,this->field_storage,this->field_type);
				this->set_death(&result);
				//printf("prod prod\n");
				//printf("%d %d %d \n",this->getRank(),b.getRank(),result.getRank());
				Prod(*this,b,result,J,normalize,alpha,true);
				return  result;
			}
			catch (hanalysis::STAError & error)
			{
				throw error;
			}
		}

		/*#############################################################
		*
		*     STA special operators
		*
		*#############################################################*/


	public:


	};




}

#define STA_FIELD hanalysis::stafield

#endif



