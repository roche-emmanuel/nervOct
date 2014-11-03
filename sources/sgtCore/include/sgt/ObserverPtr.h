#ifndef SGT_OBSERVER_PTR
#define SGT_OBSERVER_PTR

#include <sgt/RefPtr.h>
#include <sgt/Observer.h>


namespace sgt {

/** Smart pointer for observed objects, that automatically set pointers to them to null when they are deleted.
  * To use the ObserverPtr<> robustly in multi-threaded applications it is recommend to access the pointer via
  * the lock() method that passes back a RefPtr<> that safely takes a reference to the object to prevent deletion
  * during usage of the object.  In certain conditions it may be safe to use the pointer directly without using lock(),
  * which will confer a perfomance advantage, the conditions are:
  *   1) The data structure is only accessed/deleted in single threaded/serial way.
  *   2) The data strucutre is guarenteed by high level management of data strucutures and threads which avoid
  *      possible situations where the ObserverPtr<>'s object may be deleted by one thread whilst being accessed
  *      by another.
  * If you are in any doubt about whether it is safe to access the object safe then use the
  * RefPtr<> ObserverPtr<>.lock() combination. */
template<class T>
class ObserverPtr
{
public:
    typedef T element_type;
    ObserverPtr() : _reference(0), _ptr(0) {}

    /**
     * Create a ObserverPtr from a RefPtr.
     */
    ObserverPtr(const RefPtr<T>& rp)
    {
        _reference = rp.valid() ? rp->getOrCreateObserverSet() : 0;
        _ptr = (_reference.valid() && _reference->getObservedObject()!=0) ? rp.get() : 0;
    }

    /**
     * Create a ObserverPtr from a raw pointer. For compatibility;
     * the result might not be lockable.
     */
    ObserverPtr(T* rp)
    {
        _reference = rp ? rp->getOrCreateObserverSet() : 0;
        _ptr = (_reference.valid() && _reference->getObservedObject()!=0) ? rp : 0;
    }

    ObserverPtr(const ObserverPtr& wp) :
        _reference(wp._reference),
        _ptr(wp._ptr)
    {
    }

    ~ObserverPtr()
    {
    }

    ObserverPtr& operator = (const ObserverPtr& wp)
    {
        if (&wp==this) return *this;

        _reference = wp._reference;
        _ptr = wp._ptr;
        return *this;
    }

    ObserverPtr& operator = (const RefPtr<T>& rp)
    {
        _reference = rp.valid() ? rp->getOrCreateObserverSet() : 0;
        _ptr = (_reference.valid() && _reference->getObservedObject()!=0) ? rp.get() : 0;
        return *this;
    }

    ObserverPtr& operator = (T* rp)
    {
        _reference = rp ? rp->getOrCreateObserverSet() : 0;
        _ptr = (_reference.valid() && _reference->getObservedObject()!=0) ? rp : 0;
        return *this;
    }

    /**
     * Assign the ObserverPtr to a RefPtr. The RefPtr will be valid if the
     * referenced object hasn't been deleted and has a ref count > 0.
     */
    bool lock(RefPtr<T>& rptr) const
    {
        if (!_reference)
        {
            rptr = 0;
            return false;
        }

        Referenced* obj = _reference->addRefLock();
        if (!obj)
        {
            rptr = 0;
            return false;
        }

        rptr = _ptr;
        obj->unref_nodelete();
        return rptr.valid();
    }

    /** Comparison operators. These continue to work even after the
     * observed object has been deleted.
     */
    bool operator == (const ObserverPtr& wp) const { return _reference == wp._reference; }
    bool operator != (const ObserverPtr& wp) const { return _reference != wp._reference; }
    bool operator < (const ObserverPtr& wp) const { return _reference < wp._reference; }
    bool operator > (const ObserverPtr& wp) const { return _reference > wp._reference; }

    // Non-strict interface, for compatibility
    // comparison operator for const T*.
    inline bool operator == (const T* ptr) const { return _ptr == ptr; }
    inline bool operator != (const T* ptr) const { return _ptr != ptr; }
    inline bool operator < (const T* ptr) const { return _ptr < ptr; }
    inline bool operator > (const T* ptr) const { return _ptr > ptr; }

    // Convenience methods for operating on object, however, access is not automatically threadsafe.
    // To make thread safe, one should either ensure at a high level
    // that the object will not be deleted while operating on it, or
    // by using the ObserverPtr<>::lock() to get a RefPtr<> that
    // ensures the objects stay alive throughout all access to it.

    // Throw an error if _reference is null?
    inline T& operator*() const { return *_ptr; }
    inline T* operator->() const { return _ptr; }

    // get the raw C pointer
    inline T* get() const { return (_reference.valid() && _reference->getObservedObject()!=0) ? _ptr : 0; }

    inline bool operator!() const   { return get() == 0; }
    inline bool valid() const       { return get() != 0; }

protected:

    sgt::RefPtr<ObserverSet>   _reference;
    T*                          _ptr;
};

}

#endif
