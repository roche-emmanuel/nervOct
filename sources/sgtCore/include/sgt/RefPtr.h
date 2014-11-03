#ifndef SGT_REF_PTR
#define SGT_REF_PTR 1


namespace sgt {

template<typename T> class ObserverPtr;

/** Smart pointer for handling referenced counted objects.*/
template<class T>
class RefPtr
{
    public:
        typedef T element_type;

        RefPtr() : _ptr(0) {}
        RefPtr(T* ptr) : _ptr(ptr) { if (_ptr) _ptr->ref(); }
        RefPtr(const RefPtr& rp) : _ptr(rp._ptr) { if (_ptr) _ptr->ref(); }
        template<class Other> RefPtr(const RefPtr<Other>& rp) : _ptr(rp._ptr) { if (_ptr) _ptr->ref(); }
        RefPtr(ObserverPtr<T>& optr) : _ptr(0) { optr.lock(*this); }
        ~RefPtr() { if (_ptr) _ptr->unref();  _ptr = 0; }

        RefPtr& operator = (const RefPtr& rp)
        {
            assign(rp);
            return *this;
        }

        template<class Other> RefPtr& operator = (const RefPtr<Other>& rp)
        {
            assign(rp);
            return *this;
        }

        inline RefPtr& operator = (T* ptr)
        {
            if (_ptr==ptr) return *this;
            T* tmp_ptr = _ptr;
            _ptr = ptr;
            if (_ptr) _ptr->ref();
            // unref second to prevent any deletion of any object which might
            // be referenced by the other object. i.e rp is child of the
            // original _ptr.
            if (tmp_ptr) tmp_ptr->unref();
            return *this;
        }

#ifdef SGT_USE_REF_PTR_IMPLICIT_OUTPUT_CONVERSION
        // implicit output conversion
        operator T*() const { return _ptr; }
#else
        // comparison operators for RefPtr.
        bool operator == (const RefPtr& rp) const { return (_ptr==rp._ptr); }
        bool operator == (const T* ptr) const { return (_ptr==ptr); }
        friend bool operator == (const T* ptr, const RefPtr& rp) { return (ptr==rp._ptr); }

        bool operator != (const RefPtr& rp) const { return (_ptr!=rp._ptr); }
        bool operator != (const T* ptr) const { return (_ptr!=ptr); }
        friend bool operator != (const T* ptr, const RefPtr& rp) { return (ptr!=rp._ptr); }

        bool operator < (const RefPtr& rp) const { return (_ptr<rp._ptr); }


    // follows is an implmentation of the "safe bool idiom", details can be found at:
    //   http://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Safe_bool
    //   http://lists.boost.org/Archives/boost/2003/09/52856.php

    private:
        typedef T* RefPtr::*unspecified_bool_type;

    public:
        // safe bool conversion
        operator unspecified_bool_type() const { return valid()? &RefPtr::_ptr : 0; }
#endif

        T& operator*() const { return *_ptr; }
        T* operator->() const { return _ptr; }
        T* get() const { return _ptr; }

        bool operator!() const   { return _ptr==0; } // not required
        bool valid() const       { return _ptr!=0; }

        T* release() { T* tmp=_ptr; if (_ptr) _ptr->unref_nodelete(); _ptr=0; return tmp; }

        void swap(RefPtr& rp) { T* tmp=_ptr; _ptr=rp._ptr; rp._ptr=tmp; }

    private:

        template<class Other> void assign(const RefPtr<Other>& rp)
        {
            if (_ptr==rp._ptr) return;
            T* tmp_ptr = _ptr;
            _ptr = rp._ptr;
            if (_ptr) _ptr->ref();
            // unref second to prevent any deletion of any object which might
            // be referenced by the other object. i.e rp is child of the
            // original _ptr.
            if (tmp_ptr) tmp_ptr->unref();
        }

        template<class Other> friend class RefPtr;

        T* _ptr;
};


template<class T> inline
void swap(RefPtr<T>& rp1, RefPtr<T>& rp2) { rp1.swap(rp2); }

template<class T> inline
T* get_pointer(const RefPtr<T>& rp) { return rp.get(); }

template<class T, class Y> inline
RefPtr<T> static_pointer_cast(const RefPtr<Y>& rp) { return static_cast<T*>(rp.get()); }

template<class T, class Y> inline
RefPtr<T> dynamic_pointer_cast(const RefPtr<Y>& rp) { return dynamic_cast<T*>(rp.get()); }

template<class T, class Y> inline
RefPtr<T> const_pointer_cast(const RefPtr<Y>& rp) { return const_cast<T*>(rp.get()); }

}

#endif
