#include <sgtcore.h>

#include <sgt/Referenced.h>
#include <sgt/Observer.h>
// #include <sgt/DeleteHandler.h>

namespace sgt
{

//#define ENFORCE_THREADSAFE
//#define DEBUG_OBJECT_ALLOCATION_DESTRUCTION

// specialized smart pointer, used to get round auto_ptr<>'s lack of the destructor reseting itself to 0.
template<typename T>
struct ResetPointer
{
    ResetPointer():
        _ptr(0) {}

    ResetPointer(T* ptr):
        _ptr(ptr) {}

    ~ResetPointer()
    {
        delete _ptr;
        _ptr = 0;
    }

    inline ResetPointer& operator = (T* ptr)
    {
        if (_ptr==ptr) return *this;
        delete _ptr;
        _ptr = ptr;
        return *this;
    }

    void reset(T* ptr)
    {
        if (_ptr==ptr) return;
        delete _ptr;
        _ptr = ptr;
    }

    inline T& operator*()  { return *_ptr; }

    inline const T& operator*() const { return *_ptr; }

    inline T* operator->() { return _ptr; }

    inline const T* operator->() const   { return _ptr; }

    T* get() { return _ptr; }

    const T* get() const { return _ptr; }

    T* _ptr;
};

// typedef ResetPointer<DeleteHandler> DeleteHandlerPointer;
typedef ResetPointer<boost::mutex> GlobalMutexPointer;

boost::mutex* Referenced::getGlobalReferencedMutex()
{
    static GlobalMutexPointer s_ReferencedGlobalMutext = new boost::mutex;
    return s_ReferencedGlobalMutext.get();
}

// helper class for forcing the global mutex to be constructed when the library is loaded.
struct InitGlobalMutexes
{
    InitGlobalMutexes()
    {
        Referenced::getGlobalReferencedMutex();
    }
};
static InitGlobalMutexes s_initGlobalMutexes;

// static std::auto_ptr<DeleteHandler> s_deleteHandler(0);
// static DeleteHandlerPointer s_deleteHandler(0);

// static ApplicationUsageProxy Referenced_e0(ApplicationUsage::ENVIRONMENTAL_VARIABLE,"OSG_THREAD_SAFE_REF_UNREF","");

void Referenced::setThreadSafeReferenceCounting(bool enableThreadSafeReferenceCounting)
{
}

bool Referenced::getThreadSafeReferenceCounting()
{
    return true;
}

#if 0
void Referenced::setDeleteHandler(DeleteHandler* handler)
{
    s_deleteHandler.reset(handler);
}

DeleteHandler* Referenced::getDeleteHandler()
{
    return s_deleteHandler.get();
}
#endif

#ifdef DEBUG_OBJECT_ALLOCATION_DESTRUCTION
boost::mutex& getNumObjectMutex()
{
    static boost::mutex s_numObjectMutex;
    return s_numObjectMutex;
}
static int s_numObjects = 0;
#endif

Referenced::Referenced():
    _observerSet(0),
    _refCount(0)
{

#ifdef DEBUG_OBJECT_ALLOCATION_DESTRUCTION
    {
        SCOPEDLOCK(getNumObjectMutex());
        ++s_numObjects;
        logDEBUG("Object created, total num="<<s_numObjects);
    }
#endif

}

Referenced::Referenced(bool threadSafeRefUnref):
    _observerSet(0),
    _refCount(0)
{
#ifdef DEBUG_OBJECT_ALLOCATION_DESTRUCTION
    {
        SCOPEDLOCK(getNumObjectMutex());
        ++s_numObjects;
        logDEBUG("Object created, total num="<<s_numObjects);
    }
#endif
}

Referenced::Referenced(const Referenced&):
    _observerSet(0),
    _refCount(0)
{
#ifdef DEBUG_OBJECT_ALLOCATION_DESTRUCTION
    {
        SCOPEDLOCK(getNumObjectMutex());
        ++s_numObjects;
        logDEBUG("Object created, total num="<<s_numObjects);
    }
#endif
}

Referenced::~Referenced()
{
#ifdef DEBUG_OBJECT_ALLOCATION_DESTRUCTION
    {
        SCOPEDLOCK(getNumObjectMutex());
        --s_numObjects;
        logDEBUG("Object created, total num="<<s_numObjects);
    }
#endif

    if (_refCount>0)
    {
        logWARN("Warning: deleting still referenced object "<<this<<" of type '"<<typeid(this).name()<<"'");
        logWARN("         the final reference count was "<<_refCount<<", memory corruption possible.");
    }

    // signal observers that we are being deleted.
    signalObserversAndDelete(true, false);

    // delete the ObserverSet
    void* obsset = _observerSet.load(boost::memory_order_acquire);
    if (obsset) static_cast<ObserverSet*>(obsset)->unref();
}

ObserverSet* Referenced::getOrCreateObserverSet() const
{
    ObserverSet* observerSet = static_cast<ObserverSet*>(_observerSet.load(boost::memory_order_acquire));
    if (0 == observerSet)
    {
        observerSet = new ObserverSet(this);
        observerSet->ref();

        _observerSet.store(observerSet, boost::memory_order_acquire);
    }
    return observerSet;
}

void Referenced::addObserver(Observer* observer) const
{
    getOrCreateObserverSet()->addObserver(observer);
}

void Referenced::removeObserver(Observer* observer) const
{
    getOrCreateObserverSet()->removeObserver(observer);
}

void Referenced::signalObserversAndDelete(bool signalDelete, bool doDelete) const
{
    ObserverSet* observerSet = static_cast<ObserverSet*>(_observerSet.load(boost::memory_order_acquire));

    if (observerSet && signalDelete)
    {
        observerSet->signalObjectDeleted(const_cast<Referenced*>(this));
    }

    if (doDelete)
    {
        if (_refCount!=0)
            logWARN("Warning Referenced::signalObserversAndDelete(,,) doing delete with _refCount="<<_refCount);

        // if (getDeleteHandler()) deleteUsingDeleteHandler();
        // else 
        delete this;
    }
}


void Referenced::setThreadSafeRefUnref(bool threadSafe)
{

}

int Referenced::unref_nodelete() const
{
    // return --_refCount;
    return _refCount.fetch_sub(1, boost::memory_order_release)-1;
}

// void Referenced::deleteUsingDeleteHandler() const
// {
//     getDeleteHandler()->requestDelete(this);
// }

} // end of namespace sgt
