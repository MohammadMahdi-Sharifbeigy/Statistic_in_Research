set table "HW2.f13.table"; set format "%.5f"
set format "%.7e";; set samples 100; set dummy x; plot [x=0.01:5] (gamma((1+3)/2.0)/(gamma(1/2.0)*gamma(3/2.0))) * (1./3.)**(1./2.0) * (x)**(1./2.0-1.0) / ((1+(1./3.)*x)**((1+3)/2.0));
